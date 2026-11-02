use rayon::prelude::*;
use std::sync::OnceLock;
static POOL: OnceLock<rayon::ThreadPool> = OnceLock::new();
#[no_mangle]
pub unsafe extern "C" fn screen_sgemm(x: *const f32, w: *const f32, y: *mut f32, batch: usize, cols: usize, rows: usize) {
    let pool = POOL.get_or_init(|| rayon::ThreadPoolBuilder::new().num_threads(8).build().unwrap());
    let (xp, wp, yp) = (x as usize,w as usize,y as usize);
    pool.install(|| (0..8usize).into_par_iter().for_each(|worker| {
        let begin = rows*worker/8;
        let end = rows*(worker+1)/8;
        if begin == end { return; }
        unsafe { matrixmultiply::sgemm(batch, cols, end-begin, 1.0,
            xp as *const f32, cols as isize, 1,
            (wp as *const f32).add(begin*cols), 1, cols as isize,
            0.0, (yp as *mut f32).add(begin), rows as isize, 1); }
    }));
}
