-- k-means kernels. `assign` labels each point with its nearest centroid;
-- `step` runs one Lloyd iteration (assign + recompute means) and also returns
-- the largest centroid movement so the morloc driver can test convergence.
-- The tuple return `([k][d]f32, f32)` crosses the boundary as one value.

let dist2 [d] (a: [d]f32) (b: [d]f32) : f32 =
  f32.sum (map (\x -> x * x) (map2 (-) a b))

let argmin [k] (xs: [k]f32) : i32 =
  let pairs = zip xs (map i32.i64 (iota k))
  let (_, idx) =
    reduce (\(a, ai) (b, bi) -> if a <= b then (a, ai) else (b, bi))
           (f32.inf, 0i32) pairs
  in idx

entry assign [n][d][k] (pts: [n][d]f32) (cent: [k][d]f32) : [n]i32 =
  map (\p -> argmin (map (dist2 p) cent)) pts

entry step [n][d][k] (pts: [n][d]f32) (cent: [k][d]f32) : ([k][d]f32, f32) =
  let labels = map (\p -> i64.i32 (argmin (map (dist2 p) cent))) pts
  let sums =
    reduce_by_index (replicate k (replicate d 0f32))
                    (\a b -> map2 (+) a b) (replicate d 0f32)
                    labels pts
  let counts =
    reduce_by_index (replicate k 0i64) (+) 0i64 labels (replicate n 1i64)
  -- empty clusters keep their previous centroid
  let newcent =
    map3 (\s c old -> if c == 0 then old else map (\x -> x / f32.i64 c) s)
         sums counts cent
  let shift = f32.maximum (map2 (\a b -> f32.sqrt (dist2 a b)) cent newcent)
  in (newcent, shift)
