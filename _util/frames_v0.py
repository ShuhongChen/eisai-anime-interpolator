



from _util.util_v0 import * ; import _util.util_v0 as util
from _util.twodee_v0 import * ; import _util.twodee_v0 as u2d
from _util.pytorch_v0 import * ; import _util.pytorch_v0 as utorch


class DatabaseVideoFrameDuplicates:
    def __init__(
        self,
        fn_db,
        coef_lab_mean=-2.15927968008257,
        coef_lab_max=-0.20383991277453878,
        coef_intercept=4.087916207710239,
    ):
        self.fn_db = fn_db
        self.coef_lab_mean = coef_lab_mean
        self.coef_lab_max = coef_lab_max
        self.coef_intercept = coef_intercept
        
        # read data
        with sqlite3.connect(fn_db) as conn:
            c = conn.cursor()
            c.execute("""
            SELECT frame, lab_mean, lab_max FROM duplicates;
            """)
            data = c.fetchall()
        data = sorted(data, key=lambda x: x[0])
        data = np.asarray(data)
        self.data = data
        self.frames = data[:,0].astype(int)
        
        # calculate decision
        dups = (
            self.coef_lab_mean*data[:,1] +
            self.coef_lab_max*data[:,2] +
            self.coef_intercept
        )>0
        dups = np.where(dups>0)[0]
        self.duplicates = set(self.frames[dups])
        return
    def __contains__(self, frame):
        return frame in self.duplicates
    def __len__(self):
        return len(self.duplicates)

    # NOTE: gist only
    def preprocess():
        fn_video = f'{idn}/{bn}.mp4'
        fn_db = f'{odn}/{bn}.db'
        
        # setup db
        mkfile(fn_db)
        with sqlite3.connect(fn_db) as conn:
            c = conn.cursor()
            c.execute("""CREATE TABLE IF NOT EXISTS duplicates (
                frame int,
                rgb_mean float,
                rgb_max float,
                lab_mean float,
                lab_max float,
                PRIMARY KEY (frame)
            );""")
        with sqlite3.connect(fn_db) as conn:
            c = conn.cursor()
            c.execute("select max(frame) from duplicates;")
            start_frame = c.fetchall()[0][0] or 0
        start_frame = start_frame + 1
        vmd = video_metadata(fn_video)
        # if start_frame>=vmd['frame_count']:
        #     continue
        
        # get first frame
        vr0 = VideoReaderDALISeq(
            fn_video,
            bs=1,
            start=start_frame-1,
            stop=start_frame,
            step=1,
        )
        for x in vr0:
            prev = x['images']
        del vr0
            
        # process video
        vr = VideoReaderDALISeq(
            fn_video,
            bs=24,
            start=start_frame,
            step=1,
        )
        todb = []
        def _write_todb(todb):
            with sqlite3.connect(fn_db) as conn:
                c = conn.cursor()
                for vals in todb:
                    c.execute("""
                    INSERT OR REPLACE INTO duplicates (
                        frame, rgb_mean, rgb_max, lab_mean, lab_max
                    ) VALUES (
                        :frame, :rgb_mean, :rgb_max, :lab_mean, :lab_max
                    );""", vals)
            return
        for x in tqdm(vr, desc=fstrip(fn_video)):
            # batch forward
            imgs = x['images']
            frames = x['frames'].cpu().numpy()
            prevcat = torch.cat([prev, imgs[:-1]])
            diff = (prevcat - imgs).abs()
            diff_max = diff.amax((1,2,3)).cpu().numpy().astype(float)
            diff_mean = diff.mean((1,2,3)).cpu().numpy().astype(float)
            dlab = (
                kornia.color.rgb_to_lab(prevcat) -
                kornia.color.rgb_to_lab(imgs)
            ).norm(dim=1)
            dlab_max = dlab.amax((1,2)).cpu().numpy().astype(float)
            dlab_mean = dlab.mean((1,2)).cpu().numpy().astype(float)
            prev = imgs[-1:]
            for i,fr in enumerate(frames):
                todb.append({
                    'frame': int(fr),
                    'rgb_max': diff_max[i],
                    'rgb_mean': diff_mean[i],
                    'lab_max': dlab_max[i],
                    'lab_mean': dlab_mean[i],
                })

            # update db batch
            if len(todb)>10000:
                _write_todb(todb)
                todb = []
        #     if frames[-1]>10*24: break
        _write_todb(todb)
        del vr, x, imgs, prev
        gc.collect()

