let clamp (lo: i64) (hi: i64) (x: i64) : i64 = i64.max lo (i64.min hi x)

entry sobel [h][w] (img: [h][w]f32) : [h][w]f32 =
  tabulate_2d h w (\i j ->
    let px (di: i64) (dj: i64) : f32 =
          img[clamp 0 (h - 1) (i + di), clamp 0 (w - 1) (j + dj)]
    let gx = (px (-1) 1 + 2f32 * px 0 1 + px 1 1)
           - (px (-1) (-1) + 2f32 * px 0 (-1) + px 1 (-1))
    let gy = (px 1 (-1) + 2f32 * px 1 0 + px 1 1)
           - (px (-1) (-1) + 2f32 * px (-1) 0 + px (-1) 1)
    in f32.sqrt (gx * gx + gy * gy))
