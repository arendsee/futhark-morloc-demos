-- A small MLP (64 -> 32 -> 10) trained by full-batch gradient descent. One
-- `train_step` runs the forward pass, backprop, and SGD update, returning the
-- updated weights AND the batch loss as a single tuple -- computing the forward
-- pass exactly once. `predict` returns the argmax class per row. Everything is
-- matmuls, which is what Futhark is for.

let matmul [m][k][n] (x: [m][k]f32) (y: [k][n]f32) : [m][n]f32 =
  let yt = transpose y
  in map (\xr -> map (\yc -> f32.sum (map2 (*) xr yc)) yt) x

let relu2 [m][n] (x: [m][n]f32) : [m][n]f32 =
  map (\r -> map (\v -> f32.max 0f32 v) r) x

let softmax_row [n] (r: [n]f32) : [n]f32 =
  let mx = f32.maximum r
  let e = map (\v -> f32.exp (v - mx)) r
  let s = f32.sum e
  in map (\v -> v / s) e

let argmax [n] (xs: [n]f32) : i32 =
  let pairs = zip xs (map i32.i64 (iota n))
  let (_, idx) =
    reduce (\(a, ai) (b, bi) -> if a >= b then (a, ai) else (b, bi))
           (f32.lowest, 0i32) pairs
  in idx

entry train_step [b] (x: [b][64]f32) (y: [b][10]f32)
                     (w1: [64][32]f32) (w2: [32][10]f32) (lr: f32)
    : ([64][32]f32, [32][10]f32, f32) =
  let z1 = matmul x w1                 -- [b][32]
  let h = relu2 z1                     -- [b][32]
  let o = matmul h w2                  -- [b][10]
  let p = map softmax_row o            -- [b][10]
  let bf = f32.i64 b
  let loss =
    (f32.sum (map2 (\pr yr ->
                     - f32.sum (map2 (\pv yv -> yv * f32.log (pv + 1e-8f32)) pr yr))
                   p y)) / bf
  -- backprop
  let dO = map2 (\pr yr -> map2 (\pv yv -> (pv - yv) / bf) pr yr) p y  -- [b][10]
  let gW2 = matmul (transpose h) dO                                   -- [32][10]
  let dH = matmul dO (transpose w2)                                   -- [b][32]
  let dZ1 = map2 (\zr dr -> map2 (\zv dv -> if zv > 0f32 then dv else 0f32) zr dr) z1 dH
  let gW1 = matmul (transpose x) dZ1                                  -- [64][32]
  let w1' = map2 (\wr gr -> map2 (\wv gv -> wv - lr * gv) wr gr) w1 gW1
  let w2' = map2 (\wr gr -> map2 (\wv gv -> wv - lr * gv) wr gr) w2 gW2
  in (w1', w2', loss)

entry predict [b] (x: [b][64]f32) (w1: [64][32]f32) (w2: [32][10]f32) : [b]i32 =
  let h = relu2 (matmul x w1)
  let o = matmul h w2
  in map argmax o
