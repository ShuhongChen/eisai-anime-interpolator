



from _util.util_v0 import * ; import _util.util_v0 as util
from _util.twodee_v0 import * ; import _util.twodee_v0 as u2d
from _util.pytorch_v0 import * ; import _util.pytorch_v0 as utorch


############################## UTILITIES ##############################

def tvsize(p, ratio=16/9):
    w = p * ratio
    if int(w)!=w or int(p)!=p:
        warnings.warn(f'non-integer tv size')
    return (int(p), int(w))


############################## READERS ##############################

def video_timestamp(frame, fps=24):
    # converts frame number to timestamp string
    f = frame % fps
    s = int(frame / fps) % 60
    m = int(frame / fps / 60)
    return f'{m:03d}:{s:02d}+{f:02d}'
def video_metadata(fn, cap=None):
    # uses existing cap if possible
    assert os.path.isfile(fn)
    release = cap is None
    if release: cap = cv2.VideoCapture(fn)
        
    # frame count + fps
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if int(fps)!=fps:
        warnings.warn(f'fps={fps} not integer')
    if int(frame_count)!=frame_count:
        warnings.warn(f'frame_count={frame_count} not integer')
    fps = int(fps)
    frame_count = int(frame_count)
    
    # size
    size = (
        cap.get(cv2.CAP_PROP_FRAME_HEIGHT),
        cap.get(cv2.CAP_PROP_FRAME_WIDTH),
    )
    if any([int(s)!=s for s in size]):
        warnings.warn(f'size={size} not integer')
    size = tuple(int(s) for s in size)
    shape = (
        frame_count,
        3,  # assume rgb...
        *size,
    )

    # return
    if release: cap.release()
    return {
        'frame_count': frame_count,
        'fps': fps,
        'size': size,
        'shape': shape,
    }

class VideoReaderDALISeq:
    def __init__(self,
            fn, bs=24, start=0, stop=None, step=1,
            size=None, square=False, padding_mode='constant', padding_fill=0,
            dtype='float', num_threads=None,
                ):
        # metadata
        self.fn = fn
        self.bs = bs
        for k,v in video_metadata(self.fn).items():
            if k=='size':
                self.size_mp4 = v
            elif k=='shape':
                self.shape_mp4 = v
            else:
                self.__setattr__(k, v)
        
        # sequence subset
        self.start = start
        self.stop = stop if stop is not None else self.frame_count
        self.step = step
        self.file_list = [
            (self.fn, 0, self.start, self.stop),
        ]
        
        # size
        self.size = h,w = pixel_ij(size) if size!=None else self.size_mp4
        self.shape = (*self.shape_mp4[:2], *self.size)
        self.square = square
        self.padding_mode = padding_mode
        self.padding_fill = padding_fill
        s = max(h, w)
        self.padding = tv.transforms.Pad([
            (s-w)//2, # left
            (s-h)//2, # top
            (s-w)//2 + (s-w)%2, # right
            (s-h)//2 + (s-h)%2, # bottom
        ], fill=padding_fill, padding_mode=padding_mode)
        
        # misc
        self.dtype = dtype  # float: channels first, uint8: channels last
        self.num_threads = len(os.sched_getaffinity(0)) \
            if num_threads is None else num_threads
        return
    def timestamp(self, frame):
        return video_timestamp(frame, self.fps)
    def seconds(self, frame):
        return frame / self.fps
    def frame(self, s=0, m=0, h=0, f=0):
        return int((s + 60*m + 60*60*h) * self.fps + f)
    def __len__(self):
        return math.ceil(((self.stop-self.start)//self.step) / self.bs)

    class PipeIter:
        class _EmptyIter:
            def __len__(self):
                return 0
            def __next__(self):
                raise StopIteration

        @dali.pipeline_def
        def video_pipe(reader, fn_filelist):
            params = Namespace(
                file_list=fn_filelist,
                file_list_frame_num=True,
                # filenames=[ifn,],
                # labels=[0,],  # of filenames
                enable_frame_num=True,
                # enable_timestamps=True,
                device='gpu',
                name='video_reader',

                sequence_length=reader.bs,  # num frames at a time
                step=-1,             # -1=seq_len, else step
                stride=reader.step,  # btwn consecutive frames
                num_shards=1,        # partition, used for multi-gpu/node
                channels=reader.shape[1],
                dtype={
                    'float': dali.types.DALIDataType.FLOAT,
                    'uint8': dali.types.DALIDataType.UINT8,
                }[reader.dtype],
                image_type=dali.types.DALIImageType.RGB,  # RGB, YCbCr
                pad_last_batch=True,   # across shards, broken

                random_shuffle=False,  # prefetch initial_fill, then select random
                initial_fill=1024,     # for random_shuffle only

                prefetch_queue_depth=1,  # increase for cpu, gpu=1
                lazy_init=False,       # parse metadata in constructor
                normalized=False,      # ?
                preserve=False,        # prevents removal from graph
                read_ahead=False,      # amortizes large lmdb/recordio/tfrecord
                seed=-1,

                shard_id=0,            # shard index to read?
                stick_to_shard=False,  # reduces data to cache?
                skip_vfr_check=False,  # pain in ass, but might malfunction
            )
            read_node = dali.fn.readers.video(
            )
            if reader.size==reader.size_mp4:
                return tuple(dali.fn.readers.video(
                    **vars(params),
                ))
            else:
                return tuple(dali.fn.readers.video_resize(
                    **vars(params),
                    mode='stretch',
                    size=reader.size,
                ))
        def __init__(self, reader):
            self.reader = reader
            return
        def _make_filelist(self, reader):
            fl, fn_filelist = tempfile.mkstemp()
            with os.fdopen(fl, 'w') as handle:
                handle.write('\n'.join([
                    ' '.join([str(i) for i in line])
                    for line in reader.file_list
                ]))
            return fn_filelist
        def __iter__(self):
            self.fn_filelist = self._make_filelist(self.reader)
            if (self.reader.stop-self.reader.start)//self.reader.step<self.reader.bs:
                self.pipe = None
                self.dali_iter = self._EmptyIter()
            else:
                self.pipe = self.__class__.video_pipe(
                    reader=self.reader,
                    fn_filelist=self.fn_filelist,
                    batch_size=1,
                    device_id=0,
                    num_threads=self.reader.num_threads,
                )
                self.pipe.build()
                self.dali_iter = dali.plugin.pytorch.DALIGenericIterator(
                    [self.pipe,],
                    ['images', 'labels', 'frames'],
                    reader_name='video_reader',
                    # last_batch_policy=dali.plugin.base_iterator.LastBatchPolicy.FILL,
                    # last_batch_padded=True,
                )
            
            # hack for partial of last batch
            # hopefully they fix in ver>1.1.0
            tail_start = self.reader.start + \
                len(self.dali_iter)*self.reader.bs*self.reader.step
            self.tails = (self.reader.stop-tail_start) // self.reader.step
            if self.tails > 0:
                self.tail_reader = VideoReaderDALISeq(
                    self.reader.fn, self.tails,
                    start=tail_start, stop=self.reader.stop, step=self.reader.step,
                    size=self.reader.size, square=self.reader.square,
                    padding_mode=self.reader.padding_mode, padding_fill=self.reader.padding_fill,
                    dtype=self.reader.dtype, num_threads=self.reader.num_threads,
                )
                self.tail_fn_filelist = self._make_filelist(self.tail_reader)
                self.tail_pipe = self.__class__.video_pipe(
                    reader=self.tail_reader,
                    fn_filelist=self.tail_fn_filelist,
                    batch_size=1,
                    device_id=0,
                    num_threads=self.reader.num_threads,
                )
                self.tail_dali_iter = dali.plugin.pytorch.DALIGenericIterator(
                    [self.tail_pipe,],
                    ['images', 'labels', 'frames'],
                    reader_name='video_reader',
                )
            assert len(self)==len(self.reader)
            return self
        def __len__(self):
            return len(self.dali_iter) + int(self.tails>0)
        def __next__(self):
            # dali read
            try:
                try:
                    x = self.dali_iter.__next__()
                except StopIteration:
                    if self.tails>0:
                        x = self.tail_dali_iter.__next__()
                    else:
                        raise StopIteration
            except StopIteration:
                os.remove(self.fn_filelist)
                if self.tails>0:
                    os.remove(self.tail_fn_filelist)
                raise StopIteration

            # postprocess
            if self.reader.dtype=='float':
                imgs = x[0]['images'][0].permute(0,3,1,2)/255.0
            elif self.reader.dtype=='uint8':
                imgs = x[0]['images'][0]
            fr = x[0]['frames'][0,0]
            frs = fr + self.reader.step*torch.arange(
                len(imgs), device=fr.device,
            )
            if self.reader.square:
                imgs = self.reader.padding(imgs)
            
            # labels not used
            return {
                'images': imgs,
                'frames': frs,
            }
    def __iter__(self):
        return iter(self.PipeIter(self))

class VideoReaderDALIExclusion:
    def __init__(self, vr, exclude, bs=24):
        self.vr = vr
        self.exclude = exclude
        self.bs = bs
        
        # calc all streaks
        curr = []
        streak = [[], [curr,]]
        streaks = [streak,]
        cnt_streak = 0
        cnt_bs = 0
        for fr in range(self.vr.start, self.vr.stop, self.vr.step):
            # handle vr batching (reactive)
            if cnt_bs==self.vr.bs:
                curr = []
                streak[1].append(curr)
                cnt_bs = 0

            # handle non-duplicates only
            if fr not in self.exclude:            
                # add to curr
                curr.append((fr, cnt_bs)) # cnt_bs
                cnt_streak += 1

                # new streak needed (triggered)
                if cnt_streak==self.bs:
                    curr = []
                    streak = [curr, []]
                    streaks.append(streak)
                    cnt_streak = 0

            # increment vr batch
            cnt_bs += 1
        # empty tail
        if len(streaks[-1][0])==0 \
            and all([len(s)==0 for s in streaks[-1][1]]) \
            and len(streaks)>=2:
            streaks[-2][1].extend(streaks[-1][1])
            streaks = streaks[:-1]
        self.streaks = streaks
        return
    def __len__(self):
        return len(self.streaks)
    
    class FilteredIter:
        def __init__(self, dupfilter):
            self.dupfilter = dupfilter
            return
        def __len__(self):
            return len(self.dupfilter)
        def __iter__(self):
            self.vr_iter = iter(self.dupfilter.vr)
            self.vr_iter_prev = {
                'images': torch.zeros(
                    0, *self.dupfilter.vr.shape[1:],
                    device='cuda',
                ),
                'frames': torch.zeros(0, device='cuda'),
            }
            self.vr_iter_count = 0
            return self
        def __next__(self):
            # stopping condition, should raise stopiteration
            if self.vr_iter_count==len(self.dupfilter.streaks):
                return next(self.vr_iter)
            
            # read streaks
            streak = self.dupfilter.streaks[self.vr_iter_count]
            idxs_prev = [s[1] for s in streak[0]]
            idxs_next = [[s[1] for s in ss] for ss in streak[1]]
            
            # apply to ans
            images = []
            frames = []
            images.append(self.vr_iter_prev['images'][idxs_prev])
            frames.append(self.vr_iter_prev['frames'][idxs_prev])
            for idxs in idxs_next:
                ten = next(self.vr_iter)
                images_ten = ten['images']#.clone()
                frames_ten = ten['frames']#.clone()
                images.append(images_ten[idxs])
                frames.append(frames_ten[idxs])
            if len(idxs_next)>0:
                self.vr_iter_prev = {
                    'images': images_ten,
                    'frames': frames_ten,
                }
            self.vr_iter_count += 1
            return {
                'images': torch.cat(images),
                'frames': torch.cat(frames),
            }
    def __iter__(self):
        return iter(self.FilteredIter(self))

class VideoReaderCV2:
    def __init__(self, fn):
        self.fn = fn
        assert os.path.isfile(self.fn), f'video file {self.fn} not found'
        self.cap = cv2.VideoCapture(self.fn)
        for k,v in video_metadata(self.fn, cap=self.cap).items():
            self.__setattr__(k, v)
        return
    def release(self):
        return self.cap.release()
    
    def timestamp(self, frame):
        f = frame % self.fps
        s = int(frame / self.fps) % 60
        m = int(frame / self.fps / 60)
        return f'{m:03d}:{s:02d}+{f:02d}'
    def seconds(self, frame):
        return frame / self.fps
    def frame(self, s=0, m=0, h=0, f=0):
        return int((s + 60*m + 60*60*h) * self.fps + f)
    def __len__(self):
        return self.frame_count
    def __getitem__(self, idx):
        # acts just like an np.ndarray
        if isinstance(idx, int) or np.issubdtype(type(idx), np.integer):
            if idx<0: idx = len(self)+idx
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            _,frame = self.cap.read()
            ans = I(frame[...,::-1])
            return ans.np()
        elif isinstance(idx, list):
            return np.stack([self[i] for i in idx])
        elif isinstance(idx, slice) or isinstance(idx, range):
            a,b,c = idx.start, idx.stop, idx.step
            if a is None: a = 0
            if b is None: b = len(self)
            if c is None: c = 1
            if a<0: a = len(self)+a
            if b<0: b = len(self)+b
            idx = range(a,b,c)
            return np.stack([self[i] for i in idx])
        elif isinstance(idx, tuple):
            rest = ((slice(None),),())[isinstance(idx[0],int)] + idx[1:]
            return self[idx[0]][rest]
        else:
            assert 0, f'idx={idx} not understood'


############################## WRITERS ##############################

def write_animation(imgs, fn, **kwargs):
    if fn.lower().endswith('.gif'):
        return write_gif(imgs, fn, **kwargs)
    elif fn.lower().endswith('.webp'):
        return write_webp(imgs, fn, **kwargs)
    else:
        assert f'extension {fstrip(fn,1).ext} not understood'
write_ani = write_animation
def write_gif(imgs, fn, fps=1, loop=0, disposal=1):
    assert fn.lower().endswith('.gif')
    imgs = [i.pil() for i in imgs]
    dur = 1000/fps if not isinstance(fps, list) \
        else [1000/f for f in fps]
    return imgs[0].save(
        fn,
        format='GIF',
        append_images=imgs[1:],
        save_all=True,
        include_color_table=True,
        interlace=True,
        optimize=True,
        duration=dur,
        # fps
        loop=loop,
        # num times to loop, 0 forever
        disposal=disposal,
        # 0 no spec
        # 1 don't dispose
        # 2 restore to bg color
        # 3 restore to prev content
    )
def write_webp(
        imgs, fn, fps=1, loop=0,
        lossless=True, bg='k',
        quality=80, method=4,
        minimize_size=False, allow_mixed=False,
    ):
    assert fn.lower().endswith('.webp')
    imgs = [i.pil() for i in imgs]
    dur = 1000/fps if not isinstance(fps, list) \
        else [1000/f for f in fps]
    return imgs[0].save(
        fn,
        append_images=imgs[1:],
        save_all=True,
        duration=int(dur),
        loop=loop,     # 0 forever
        lossless=lossless,
        quality=quality,  # 1-100
        method=method,  # 0-6 (fast-better)
        background=c255(bg),
        minimize_size=minimize_size,  # slow write
        allow_mixed=allow_mixed,  # mixed compression
    )
def copy_video_audio(source, target, postfix='_audio'):
    fs = fstrip(target, return_more=True)
    ofn = f'{fs["dn"]}/{fs["bn"]}{postfix}.{fs["ext"]}'
    sp = subprocess.run([
        '/usr/bin/ffmpeg',
        '-i', target,
        '-i', source,
        '-c:v', 'copy',
        '-map', '0:v:0',
        '-map', '1:a:0',
        '-c:a', 'aac',
        '-b:a', '192k',
        ofn,
    ])
    assert sp.returncode==0
    return ofn

class VideoWriterCV2:
    def __init__(self, fn, fps=24, overwrite=True):
        self.fn = fn
        self.fps = fps
        self.overwrite = overwrite
        self.initialized = False
        self.size = None
        self.writer = None
        self.frame_count = 0
        return
    def write(self, frame):
        # initialize if not already
        if not self.initialized:
            if os.path.isfile(self.fn):
                if self.overwrite:
                    os.remove(self.fn)
                else:
                    assert 0, f'{self.fn} already exists'
            self.size = frame.size
            self.writer = cv2.VideoWriter(
                self.fn,
                cv2.VideoWriter_fourcc(*'MP4V'),
                self.fps,
                self.size[::-1],
            )
            self.initialized = True
            
        # write and update
        # for some reason, must convert to rgb before cv2
        if frame.size!=self.size:
            warnings.warn(f'frame.size={frame.size} != self.size={self.size}')
        self.writer.write(frame.convert('RGB').cv2())
        self.frame_count += 1
        return
    def release(self):
        self.writer.release()
        return


