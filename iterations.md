# Original dataset

- Ampere 0, pmf imputation for allergens, mse - 1. \* f1
- Ampere 1, knn imputation for allergens, mse - 1. \* f1
- Citrus 0, pmf imputation, mse - 1. \* f1
- Bastet 0, pmf imputation, mse - 1. \* f1

PMF imputation mse - 1. \* f1

- Ampere 2, full sigmoid (except conv branch and last layer)
- Ampere 3, tanh (linear) / gelu (conv) / sigmoid (last)
- Ampere 4, gelu (linear) / gelu (conv) / sigmoid (last)

Linear/Conv with GELU activation

- Dexter 0, latent dim 128
- Ampere 5, loss = bce
- Ampere 6, loss = bce - 1. f1
- Ampere 7, bp_mll_loss
- Ampere 8, deeper central branch, loss = bce - 1. f1
- Ampere 9, deeper central branch, loss = mse - 1. f1

Linear/Conv with SILU activation

- Ampere 10, loss = mse - 1. f1, back to shallower design
- Ampere 11, max/avg pooling
- Ampere 12, same (more patience)
- Ampere 13, less channels in conv branch
- Ampere 14, Images have 1 channel
- Ampere 15, Image normalization (0, 1), 3 channels
- Ampere 16, Image scaling + padding-resize
- Ampere 17, pad, norm, Aggressive downsizing
- Ampere 18, same but with Conv2d instead of resnet
- Ampere 19, global allergen scaling (rather than per-column scaling)
- Ampere 20, whole train dataset
- Ampere 21, blind -> Garbage
- Ampere 22, deaf (no tabular data fed) -> Garbage
- Euclid 0
- VAE 1: resnet encode/decode architecture
- Farzad 1, resnet vae
- Ampere 23, pad resize (512) normalize
- Ampere 24, same but with deep maxpools, dropout

90/10 train/val split

- Ampere 25, like 17
- Gordon 0
- Gordon 1, pure bce loss
- Ampere 25, weighted bce loss
- Gordon 2, weighted bce loss
- Gordon 3, rebalanced bce loss
- Ampere 26, rebalanced bce loss
- Gordon 4, rebalanced bce loss with fine hyperparameters

SCALE-PAD-RESIZE(128x128) IMAGE PREPROC., NO BIAS IN CONV LAYERS

- Gordon 5, focal loss
- Gordon 6, distribution balanced loss
- Gordon 7, rebalanced_binary_cross_entropy_with_logits
- Gordon 8, focal_loss_with_logits
- Ampere 27, rebalanced_binary_cross_entropy_with_logits
- Ampere 28, bce - f1 => .58

SCALE-PAD-RESIZE(512x512) IMAGE PREPROC., NO BIAS IN CONV LAYERS

- Ampere 29, focal_loss_with_logits => .62
- Ampere 30, bce - f1 => .62
- Ampere 31, rebalanced_binary_cross_entropy_with_logits => .581
- Gordon 9, focal_loss_with_logits => .679
- Gordon 10, bce - f1 => 0.677
- Gordon 11, rebalanced_binary_cross_entropy_with_logits => 0.673
- Ampere 32, mse loss => .6
- Gordon 12, mse loss => .685
- Gordon 13, rebalanced_binary_cross_entropy_with_logits - f1 => .675
- Gordon 14, focal_loss_with_logits - 2 \* f1 => 0.676
- Helena 0, focal, nds=5, n_a=n_d=128, gamma=1.5, nef=n_d, ls=5e-3
- Helena 1, focal, nds=10, n_a=n_d=32, gamma=1.5, nef=256, ls=5e-3, NaNs after
  ~700 steps
- Helena 2, focal, nds=3, n_a=n_d=32, gamma=1, nef=32, ls=5e-3, no initial
  convolution (just norm + maxpool) => .587
- Helena 3, focal, nds=3, n_a=n_d=16, gamma=1.5, nef=16, ls=1e-3, no initial
  convolution (just norm + maxpool) => NaN
- Helena 4, bce, nds=3, n_a=n_d=16, gamma=1.5, nef=16, ls=1e-3, some fixes, all
  relu => bad at training
- Helena 5, vision encoder with actual resnet steps, decision aggregation is
  just reshape, bce, nds=3, n_a=n_d=16, gamma=1.5, nef=16, ls=1e-3, all relu =>
  NaN
- Helena 6, vision encoder with actual resnet steps, decision aggregation is
  just reshape, bce, nds=4, n_a=n_d=128, gamma=1.5, nef=128, ls=1e-3, lr=1e-4
  all relu => bad at training
- Helena 7, vision encoder with actual resnet steps, decision aggregation is
  just SUM (i.e. encoded features in fusion layers, like in Gordon), bce,
  nds=4, n_a=n_d=128, gamma=1.5, nef=128, ls=1e-3, lr=1e-3 all relu
- Helena 8, nds=5, n_a=32, n_d=256, gamma=1.5, nef=256, ls=1e-5, lr=1e-3 => NaN

Sparse loss is numerically unstable (because of the log ><)
Sparse loss is not 2-norm

- Helena 9, nds=3, n_a=32, n_d=256, gamma=1.5, nef=256, ls=0
- Helena 10, nds=3, n_a=32, n_d=32, gamma=1.5, nef=32, ls=0, lr=1e-4, simple
  convolution layers in vision encoder
- Helena 11, nds=3, n_a=256, n_d=256, gamma=1.5, nef=256, ls=1e-5, lr=1e-4,
  simple convolution layers in vision encoder
- Helena 12, nds=5, n_a=128, n_d=128, gamma=1.5, nef=128, ls=0, lr=1e-4,
  simple convolution layers in vision encoder, bce loss
- Helena 13, nds=5, n_a=64, n_d=64, gamma=1.5, nef=64, ls=0, lr=1e-4,
  resnet conv in vis. enc., bce loss
- Helena 14, nds=3, n_a=64, n_d=64, gamma=1.5, nef=64, ls=0, lr=1e-4,
  resnet enc in vis. enc., bce loss, batch size at 128
- Helena 15, nds=3, n_a=64, n_d=64, gamma=1.5, nef=64, ls=0, lr=1e-4,
  resnet conv. in vis. enc., bce loss
- Helena 16, nds=3, n_a=64, n_d=64, gamma=2, nef=64, ls=5e-3, lr=1e-4,
  resnet conv. in vis. enc., focal loss
- Helena 17, nds=3, n_a=64, n_d=64, gamma=2, nef=64, ls=5e-3, lr=1e-4,
  resnet conv. in vis. enc., db loss
- Helena 18, nds=5, n_a=512, n_d=512, gamma=2, nef=256, ls=1e-12, lr=1e-3,
  db loss

With large n_a and n_d the sparse loss goes weeeeee. Switching from
e = -m \* (m + 1e-10).log()
return e.sum(-1).mean() # Mean row entropy
to
return e.mean() # (global) mean entropy

- Helena 19, nds=5, n_a=512, n_d=512, gamma=2, nef=256, ls=1e-12, lr=1e-3,
  focal
- Helena 20, nds=5, n_a=1024, n_d=1024, gamma=2, nef=256, ls=1e-3, lr=1e-3,
  focal

512 ~ N_FEATURES performs better than any other choice for n_a, n_d

- Helena 21, nds=5, n_a=256, n_d=256, gamma=2, nef=256, ls=1e-3, lr=1e-3,
  focal
- Helena 22, nds=5, n_a=512, n_d=512, gamma=2, nef=256, ls=1e-3, lr=1e-3,
  focal

Actually it's the sparse loss term. Clamping sparse loss to -100, 0

- Helena 23, nds=5, n_a=512, n_d=512, gamma=2, nef=256, ls=1e-3, lr=1e-3,
  focal
- Helena 24, nds=5, n_a=512, n_d=64, gamma=2, nef=64, ls=0, lr=1e-3,
  focal => ALL 0's lolwhat
- Helena 25, nds=5, n_a=512, n_d=512, gamma=2, nef=256, ls=1e-12, lr=1e-3,
  db loss
- Helena 26, nds=5, n_a=1024, n_d=1024, gamma=2, nef=256, ls=1e-3, lr=0,
  db loss

DB loss gives all 0s?

- Helena 27, nds=5, n_a=512, n_d=512, gamma=2, nef=256, ls=1e-3, lr=1e-3,
  focal loss
- Helena 28, nds=5, n_a=512, n_d=512, gamma=2, nef=256, ls=5e-3, lr=1e-3,
  focal loss
- Gordon 15, focal, initial max pooling (no conv.)
- Gordon 16, vision encoder with fusion after last conv
- Ingrid 0, focal loss, resnetconv, 32x32 switch to vision encoder, qkv=vuu
- Ingrid 1, focal loss, resnetconv, 32x32 switch to vision encoder, qkv=uvv
- Jackal 0, focal, 128 latent features
- Jackal 1, focal, 64 latent features, resnet linear in last branch
- Jackal 2, focal, 64 latent features, no resnet linear in tabular branch
- Jackal 3, balanced, 64 latent features, no resnet linear in tabular branch
- Jackal 4, bce - f1, 64 latent features, no resnet linear in tabular branch => .659
- Jackal 5, rebalanced, 64 latent features, no resnet linear in tabular branch
- Jackal 6, 256 l.feats, maxpool+samesizeConv in initial pooling

# NEW DATASET

FOCAL LOSS

- Kadgar 0, 256 feats, maxpool/2 initial pooling
- Jackal 0, 256 feats, maxpool/2 + sameconv initial pooling
- Gordon 0,
- Jackal 1, no maxpool, conv/2
- Jackal 2, resnet/2, deeper linear chains
- Jackal 3, maxpool/2, resnet vision encoder, deeper linear chains => bad
- Gordon 1, resnet basic layer vision encoder => same as Gordon 0
- Gordon 2, same, SGD/OneCycleLR optimization => terrible
- Gordon 3, resnet stage in vision encoder => 0.246

- Gordon 4, same => 0.243
- Jackal 4, resnet stage vision encoder
- Gordon 5, rebalanced_bce_with_logits => 0.247

FIXED distribution_balanced_loss_with_logits - f1

- Gordon 6
- Gordon 7, relu => .252
- Gordon 8, gelu => .248 but the f1 on training is really good!
- London 0, mean f1 .625696 => .2519
- London 1, res lin blk. => mean f1 .6957,
- Jackal 5, => mean f1 .527
- Jackal 6, pairwise chan diff => mean f1 .4268
- London 2, deeper main branch, ps=16, nt=16, nh=16 => mean f1 .442
- London 3, deeper main branch, ps=16, nt=8, nh=8 => mean f1 .4258

FOCAL

- London 4, ps=16, nt=8, nh=8, d=.2 => mean f1 .5787 | .2530
- Gordon 9, more hidden channels
- Gordon 10, truncated vision encoder => .536 .25
- Gordon 11, db loss, truncated vision encoder => .474 .2458
- Gordon 12, wbce prevmax/prev => .4392 .2472
- Gordon 13, wbce effective n samples => .310
- London 5, wbce effective n samples => .282
- London 6, wbce prevmax/prev => .367789
- London 7, focal, revised arch => .550
- London 8, cb-focal, revised arch
- London 9, focal, nn.TransformerEncoderLayer => .4516 .2494
- London 10, db loss, nn.TransformerEncoderLayer => .570886

KNN imputation

- London 11, db loss => .7281
- London 12, focal => .7104
- Gordon 14, db loss => .66
- London 13, mlsmote 1, db loss => .7585
- London 14, mlsmote 1, focal => .7644
- London 15, mlsmote 1 ONLY ON TRAIN DS, focal => .723995

# NEW DATASET (again)

- London 0: knn, mlsmote, db loss => .8222 .331
- London 1: same but fixed mlsmote =>
- London 2: Smaller arch, weight decay, .1 drop, db => val/loss increase

TWEAKED REMEDIAL-MLSMOTE

- London 3: original arch .1 drop, bce => val/loss increase
- London 4: db loss

NO MLSMOTE

- London 5: gated vision branch
- London 6: gated vision branch, bce with irlbl weights
- London 7: same, edim=128, drop=.5 (in main branch) => slight overfit. .49 ...
- London 8: same, onecyclelr => overfit, .52
- London 9: onecyclelr, main drop=.25, ve drop=.1, edim=256
- London 10: main drop=.5, ve drop=.5, edim=256
- London 11: main drop=.5, ve drop=.5, edim=256
- London 12: oversampling, main drop=.25, ve drop=.25, edim=256

NO REDUCELRONPLATEAU, no oversampl, no nandrop (ytrue repl)

- London 13: main drop=.5, ve drop=.5, edim=256 => local and trustii f1 are
  close: .55
- Gordon 0: smaller arch, main drop = .5, edim = 256, trunc ve => trustii .58
- London 14: onecyclelr
- Gordon 1: onecyclelr => .56
- London 15: all drop .25 => .61 but still overfitting
- London 16: no drop, edim = 256 => .65
- London 17: drop .5, edim 128, 4 trans, 1 head => .56
- London 18: edim 512, 16 trans, 8 head => .5589
- London 19: same, no drop => .6672
- London 20: same, no drop, db loss => .67
- Masala 0: method a, edim 512, no drop, db loss => .65
- Masala 1: method b, edim 256, nt 16, nh 16, no drop, db loss => .66
- Masala 2: method c, edim 256, nt 16, nh 16, no drop, db loss =>
- Masala 3: method f, edim 256, nt 4, nh 4, .25 drop, wbce loss => bad
- Masala 4: method a, edim 1024, nt 32, nh 16, .1 drop, wbce loss => bof
- London 21: edim 1024, nt 32, nh 16, .1 drop, wbce loss =>
- London 22: edim 1024, nt 16, nh 8, .2 drop, wbce loss => .6
- London 23: noise + full penc, no pooling, ps 32, edim 1024, nt 16, nh 8, .5
  drop, wbce loss => .569
- London 25: no noise, ps 32, no learnable pe, main drop .2
- London 26: no noise, ps 32, no learnable pe, no pooling, ps 32, edim 64, nt
  16, nh 8, main drop .5 =>
- London 27: same but with onecyclelr
- London 28: same but with vit-pytorch
- London 29: same but with huggingface transformers

# ADDED TRUE TARGETS

ONE CYCLE LR

- London 0: ps=32, edim=64, nt=16, ng=8, mdrop=.5, homemade transformer => .55
- London 1: ps=32, edim=512, nt=16, ng=8, mdrop=.5 => .55
- London 2: ps=32, edim=512, nt=16, ng=8, mdrop=0 => .54
- London 3: ps=32, edim=512, nt=16, ng=8, mdrop=0, db loss => .58
- London 4: pooling, ps=16, edim=512, nt=16, ng=8, mdrop=.5, db loss => .52
- London 5: no noise. no posenc. ps=16, edim=512, nt=16, ng=8, mdrop=.25, db
  loss =>

NO EARLY STOPPING => ONE CYCLE LR

- London 6: no noise. no posenc. ps=16, transdim=128, embed=4\*td, nt=16, nh=16,
  mdrop=.5, db loss, mult-merge =>

CONSTANT LR

- London 7: no noise, no posenc, ps=32, edim=512, nt=16, nh=16, ndrop=.5, db
  loss, concat-merge =>
- London 8: ps=32, edim=512, nt=16, nh=8, mdrop=.1, db loss =>
- London 9: same, smaller lr => .6
- London 10: ps=32, edim=512, nt=16, nh=8, mdrop=.5, vtdrop=.1 => .54
- London 11: ps=16, edim=512, nt=8, nh=8, mdrop=.5 =>

ONECYCLELR

- Gordon 0: all 8s channels, edim=64, mdrop=.5 => .52
- Gordon 1: edim=512, mdrop=.5 => terrible
- Gordon 2: all 4s, edim=32, mdrop=.25 =>
- Gordon 3: all 4s, pooling, edim=16, mdrop=.1, mult comb => .57
- Gordon 4: same, wbce => .57
- Gordon 4: same, focal => .58
- London 13: edim=32, ps=16, nt=8, ng=8, mdrop=.5, pool, wbce => .511
- London 14: edim=64, ps=16, nt=8, ng=8, mdrop=.5, pool, db loss => .52
- London 15: edim=64, ps=16, nt=8, ng=8, mdrop=0, pool, wbce =>
- Masala 0: a, edim=64, ps=8, pt=8, mdrop=0, pool, db loss => .6
- Masala 1: same, mdrop=.1, wbce => .57
- Masala 2: b, edim=64, ps=8, pt=8, mdrop=0, pool, db loss => .57
- Masala 3: a, edim=256, ps=8, pt=8, mdrop=0, pool, db loss => .57
- London 16: edim=64, ps=16, nt=8, ng=8, mdrop=0, pool, db loss, no output
  correction => local .34
- London 17: same, output correction => local .344

MLSMOTE WITHOUT REMEDIAL, ONECYCLELR

- London 18: same => .63
- London 19: same, impute targets => local .5 .61
- Norway 0: ed=64, ps=16, nt=8, nh=8, drop=.1 => local .61 .64
- Norway 1: ed=1024, ps=16, nt=8, nh=8, drop=.1 => local .59
- Norway 2: ed=1024, ps=16, nt=16, nh=16, drop=.25 => local .636 but nan loss
  at the end
- Norway 3: same, constant lr=1e-3 => nan loss
- Norway 4: 7 maxpool, same, constant lr=1e-4 => local .62 .648

# PROPER HPARAMS BOOKKEEPING

MAXPOOL 7

- Norway 0: const. lr=1e-4 edim=128, ps=16, nt=32, nh=8, dr=.1 => local .585

REDUCE LR ON PLATEAU val/loss

- Norway 1: edim=64, ps=16, nt=32, nh=4, dr=.1 => local .539
- Norway 2: edim=1024, ps=16, nt=8, nh=8, dr=.1 => local .543

DATAMODULE + ALL 2 GPUs

- Norway 3: same as Norway 1
- Norway 4: edim=32, ps=16, nt=64, nh=4, dr=.1 => loc .5 .62
- Norway 5: same, deeper arch
- Norway 6: triple fusion, edim=32, ps=16, nt=16, nh=8, dr=.1
- Norway 7: following hparam search 1, edim=512, ps=16, nt=16, nh=8, dr=.1 =>
  .64
- Norway 8: same, early stopping on val/f1
- Norway 9: val/loss, more patience, no sched => .724 .6583

IMAGE SIZE 256, NO POOLING, ReduceLROnPlateau val/f1

- Norway 10: edim=512, ps=16, nt=16, nh=8, dr=.25, wd=1e-1, sched, double
  merge =>
- Norway 11: edim=512, ps=16, nt=16, nh=8, dr=.25, no wd, =>
- Norway 12: edim=512, ps=16, nt=16, nh=8, dr=.1, no wd, => .75 .639
- Norway 13: same, wd=1e-1 => .634 .635
- Norway 14: same, wd=1e-3, wbce => .72 .628

IMAGE SIZE = 128, NO POOLING, ReduceLROnPlateau val/f1

- Norway 17: same, wd=0, db loss => .75 .645
- Norway 18: same, wd=1e-3 => loc .73
- Norway 19: same, wd=1e-1 => loc .65
- Norway 20: same, iges imputation by most frequent => .69 .666
- Norway 21: same, iges imputation by most frequent, no oversampl, no
  dropnantgt, no imputetgt => loc .517988

IGES IMPUTATION BY MOST FREQUENT, WD=1e-1, DB LOSS LS=1e-4, IMGSIZE=128

- London 0: edim=512, ps=16, nt=16, nh=8, dr=.1, no pool => loc .707
- London 1: same, mlsmote sampling factor = 10 => loc .681 .6471

- Norway 22: edim=512, ps=16, nt=16, nh=8, dr=.1, mlsmote without drapnatgt =>
  meh
- Norway 23: same, global KNN imputation
- Norway 24: same, global KNN imputation, dropnan, oversample => .72 .63
- London 2: same => .743 .65
- London 3: edim=512, ps=8, nt=16, nh=8, dr=.25 => .7592 .6384
- Norway 25: edim=512, ps=8, nt=16, nh=8, dr=.1 => .7118 .641
- London 4: edim=512, ps=8, nt=16, nh=8, dr=.5, tab branch with tanh => .729
  .629
- Norway 26: edim=512, ps=8, nt=16, nh=8, dr=.2, tab branch with tanh, blind =>
  .68 .63
- London 5: edim=512, ps=8, nt=16, nh=8, dr=.5, blind =>

QUICK&DIRTY KNN FIT

- Norway 27: more drops, activation in coattvit => .718 .64739
- London 6: more drops, d=.1, onecyclelr =>
- London 7: more drops, d=.1, lr=1e-3, wd=1e-2 => nan loss
- London 8: more drops, d=.1, lr=1e-4, wd=1e-2 =>
- Orchid 0: tab(edim=32, nt=4, nh=8), vit(edim=512, nt=16, nh=8), d=.1
  => .692373 .663
- Orchid 1: tweaked head (main branch), tab(edim=512, nt=16, nh=8),
  vit(edim=512, nt=16, nh=8), d=.1 => .766 .665
- Orchid 2: same, d=.1, wd=1e-1 =>
- Orchid 3: d=.1, wd=1e-3, tab(edim=32, nt=32, nh=16), vit(edim=32, nt=32,
  nh=16), mlp_dim=2048 => loc. .75745
- London 9: huggingface vit, d=.1 => overfit
- London 10: frozen pretrained vit, d=.1 => overfit .7977 .658
- London 11: same, d=.5 => slight overfit loc. .748564

IMAGE SIZE = 256

- London 12: same, d=.5, wd=5e-3, =>
- London 13: frozen pretrained vit, d=.5, wd=5e-3 =>
- London 13: pretrained vit (not frozen), d=.5, wd=5e-3, no sched =>
- Orchid 4: frozen vit, tab(edim=512, nt=16, nh=8), d=0, wd=5e-3, no sched =>
- Orchid 5: frozen vit, tab(edim=512, nt=16, nh=8), d=.1, wd=5e-3, no sched =>

NO SCHED

- Orchid 6: frozen vit, tab(edim=512, nt=16, nh=8), d=.1, swa=1e-4 =>
- Orchid 7: same, d=.1, swa=1e-3 =>
- Orchid 8: same, d=0, wd=1e-3, swa=1e-3 =>
- Orchid 9: same, d=0, wd=1e-3, swa=1e-3, wbce =>
- Orchid 10: live vit, d=0, lr=5e-4, swalr=1e-3, swae=10, wd=0, BCE =>
- Norway 28: edim=512, ps=16, nt=16, nh=8, dr=0, BCE => .805 .64869
- Norway 29: same, BD loss =>
- Orchid 11: same, d=0.5, wd=1e-3, db =>
- Orchid 12: same, d=0.2, wd=1e-3, db =>

ITERATIVE IMPUTATION

- Orchid 13: lr=5e-4, wd=1e-3, d=.1, no swa, db =>
- Orchid 14: frozen, lr=1e-4, wd=5e-3, d=.1, reducelr, db =>
- Orchid 15: same, vit not frozen => vit learns, .68
- Orchid 16: same, no padding => .77299 .67
- London 14: pretrained vit (not frozen), edim=512, drop=.5, wd=5e-3 =>
- Orchid 17: same as 16, .5.5 normalized images =>

NO OVERSAMPLING

- Orchid 18: same, wd=1e-2, drop=.1 => overfitting .82 .68
- Orchid 19: wd=1e-2, drop=.5, edim=64, mlp=2048 => .36
- Orchid 20: wd=1e-2, drop=.5, edim=512, mlp=2048 =>

- Orchid 21: wd=1e-2, drop=.1, edim=512, mlp=2048 => .61 .69

- Orchid 22: same, no over, no dropnan => .777 .687
- Orchid 23: same, no over, no dropnan, onecyclelr => .69 .67
- Orchid 24: same, no over, no dropnan, onecyclelr, 80/20 split =>
- Orchid 25: same, 100 epochs, no early stopping =>
- London 15: pretrained vit (not frozen), edim=512, drop=.1, wd=1e-2, no over,
  no dropnan, early stopping, 80/20 split => overfit
- London 16: same, wd=.1 => overfit
- Norway 30: d=.1, edim=768, nt=12, nh=12, ps=16 => overfit
- Norway 31: d=.1 in vit too, edim=768, nt=12, nh=12, ps=16 => overfit
- Norway 32: d=.1 in vit too, edim=512, nt=8, nh=8, ps=16 => overfit
- Norway 33: d=.1 in vit too, edim=64, nt=8, nh=8, ps=16 => overfit

- London 17: edim=128, d=.1, wd=1e-2, google/vit-base-patch16-224 instead of
  google/vit-base-patch16-224-in21k =>

- Orchid 26: wd=1e-2, drop=.1, edim=512, mlp=2048, hierarchical logit
  correction => .53 .677
- Orchid 27: same but no correction => .75 .691

- Orchid 28: wd=1e-2, drop=.1, edim=512, mlp=2048, no logit correction,
  dropnan, .9 split => .6935 .692
- Orchid 29: same but with correction => .4486 .64

- Orchid 30: wd=1e-2, drop=.1, edim=512, mlp=2048, no logit correction,
  dropnan, .5 split => .46 .64

- Orchid 31: wd=1e-2, drop=.1, edim=512, mlp=2048, no dropnan, .9 split, mc
  loss => .4233 .56

EARLY STOPPING ON VAL/LOSS (insteat of val/f1)

- Orchid 32: wd=1e-2, drop=.1, edim=512, mlp=2048, no dropnan, .8 split, logit
  correction (min), db loss => .2517 .489
- Orchid 33: same, logit correction (max), db loss => .42 .54
- Orchid 34: same, logit correction (max), mc loss => .549 .4933
- Orchid 35: same, logit correction (max), bce loss => .32

- London 18: wd=1e-2, drop=.1, conf like google vit, no dropnan, .8 split, logit
  correction (max), mc loss => .6363 .5877
- London 19: logit correction (max), db loss => .89 .65

- Primus 0: ced=16, ed=512, d=.1, no correction, db loss => overfit .883 .6490
- Primus 1: ced=8, ed=256, d=.2, no correction, db loss => overfit .86
- Primus 2: ced=16, ed=128, d=.5, no correction, db loss => .183
- Primus 3: ced=16, ed=256, d=.1, wd=1e-1, no correction, db loss => .253
- Primus 4: ced=16, ed=256, d=.1, wd=5e-2, no correction, db loss => .816 .65
- Primus 5: ced=16, ed=2048, d=.1, wd=5e-2, no correction, db loss => .88
- Primus 6: same, d=.1 but .5 on mlp head => .88 .65

- Orchid 36: TabTransformer mlp head is a layers.MLP, wd=1e-2, drop=.1,
  edim=512, mlp=2048, no corr., db loss, no norm in MLPs =>
- Orchid 37: same, TabTransformer mlp no layer norm, norm in MLPs =>
- Orchid 38: same, wd=1e-3 =>
- Orchid 39: same, revert TabTransformer MLP, wd=1e-2 =>
- Orchid 40: same, TabTransformer mlp no layer norm, wd=1e-2 =>

ITERATIVE IMPUTER (separate since can't serialize), STILL SCHED ON val/loss

- Orchid 41: edim=512, nt=16, nh=8, d=.1, mlpd=2048, wd=1e-2, db loss,
  TabTransformer mlp is layers.MLP with no layer norm =>
- Orchid 42: edim=512, nt=16, nh=8, d=.1, mlpd=2048, wd=1e-2, db loss,
  TabTransformer mlp is layers.MLP WITH layer norm =>

80/20 SPLIT, KNN, MAXPOOLING

- Orchid 43: same, d=.2, layers.MLP with layer norm =>
- Orchid 44: same, d=.1 =>
- Orchid 45: same, d=.5 =>
- Orchid 46: same, d=.3 =>
- Orchid 47: same, d=.25 =>

SCHED/ES on val/f1 with tol=1e-2

- Orchid 48: same, d=.25 => .825 .68
- Orchid 49: same, d=.1 => .859 .677
- Orchid 50: same, d=.1, bce loss =>
- Orchid 51: same, d=.25, db, max-correction =>
- Orchid 52: same, d=.25, no correction, padded images => .829 .684

- London 20: edim=512, d=.1, wd=1e-2, no corr, padimg =>
- London 21: same, d=.5 => .747 .66
- London 22: same, edim=256, d=.25, mlp=512, pool => .802 .67
- London 23: edim=128, mlp=256, d=.1, wd=5e-2 pool =>
- London 24: same, swa 1e-2 5 => .394
- London 25: same, swa 1e-2 5, early on val/loss =>
- London 26: same, swa 1e-3 10, no img pad&norm =>
- London 27: same, wd=0 => .8542 .67
- London 28: same, wd=1e-3 =>
- London 29: same, swa=1e-4 10, wd=1e-3 =>
- London 30: edim=64, mlpd=128, d=.1, wd=1e-3 => .79
- London 31: edim=32, mlpd=64, d=.1, wd=1e-3 =>
- London 32: edim=32, mlpd=64, d=.25, wd=1e-3 =>
- London 32: edim=32, mlpd=64, d=.25, wd=1e-3, lr-1e-3 =>

- Orchid 53: edim=512, mlp=2048, d=.1, pooling, rl=1e-4, wd=1e-3, swa 1e-3 10
  =>
- Orchid 54: edim=512, mlp=2048, d=.1, pooling, rl=1e-4, wd=1e-3, swa 1e-4 10
  =>
- Orchid 55: edim=128, mlp=512, d=.1, pooling, rl=1e-4, wd=1e-3, swa 1e-4 10,
  still no pad, .5 .5 norm, .05 noise, .7 ratio =>
- Orchid 56: same, edim=256 =>
- Orchid 57: same, edim=256, d=.25 =>
- Ampere 0: .77 .6
- Norway 37: edim=512, nt=8, nh=4, d=.1, wd=1e-3, pooling, db loss => .84 .65
- Norway 38: edim=256, mlp=1024, nt=8, nh=4, d=.1, wd=1e-3, pooling, bce loss
  =>
- Norway 39: same, nt=16, nh=8, bce => .72 .63
- Norway 40: edim=512, mlp=2048, nt=16, nh=8, swa 1e-4 10, bce => .79 .64
- Norway 41: edim=512, mlp=2048, nt=16, nh=8, swa 1e-2 10, bce =>
- Norway 42: onecycle, bce =>
- Norway 43: onecycle, focal => .79 .65
- Orchid 58: onecycle, focal =>
- London 34: edim=64, mlp=512, d=.25, wd=1e-3, onecycle, no early stopping, db
  loss =>
- London 35: same, cosine annealing sched =>
- London 36: edim=64, mlp=512, d=.25, wd=1e-2, reduce lr val/ham =>
- London 37: edim=64, mlp=512, d=.25, reduce lr val/ham, irlblbce =>
- London 38: edim=64, mlp=512, d=.25, reduce lr val/f1, bce =>
- London 39: edim=64, mlp=512, d=.1 =>
- London 40: edim=64, mlp=512, d=.1, wd=1e-3 =>
- London 41: edim=64, mlp=256, d=.1, wd=0, lr=1e-3 =>
- London 42: edim=64, mlp=256, d=.2, wd=0, lr=1e-4 =>

- Orchid 59: edim=64, mlp=256, d=.1, wd=0, lr=1e-3 =>
- Orchid 60: same, bce loss =>

- Orchid 61: edim=512, mlp=2048, d=.1, wd=1e-2, lr=1e-3, bce => .38 .64
- Orchid 62: same, iterative imputation, bce => .37 .63

70% SPLIT, still KNN, BCE LOSS, LR=1e-4

- Orchid 63: same => .62 .65
- Orchid 64: same, lr=1e-4, db =>
- Orchid 65: same, lr=1e-4, wd=5e-2, db =>
- Orchid 66: same, wd=5e-2 => .45 .645
- Orchid 67: same, wd=0 => .68 .666
- Orchid 68: same, wd=1e-3, irlbl_bce => .41 .56
- Orchid 69: same, wd=1e-3, irlbl_bce, swa 1e-4 20 => .567 .6
- Orchid 70: same, wd=1e-3, swa 1e-4 20 => .75 .672
- Orchid 71: same, wd=1e-2, swa 1e-4 20 => .624 .66
- Orchid 72: ed=256, mlp=1024, d=.1, wd=1e-3, swa 1e-4 20 => .68
- Orchid 73: ed=1024, mlp=1024, d=.1, wd=1e-3, swa 1e-4 20 => .67
- Orchid 74: ed=512, mlp=4096, d=.1, wd=1e-3, swa 1e-4 10 => .61
- Orchid 75: 512/512, d=.1, wd=1e-3, swa 1e-4 20 => .61
- Orchid 76: 512/512, d=.1, wd=1e-3, no swa =>
- Orchid 77: 512/512, d=.1, wd=1e-3, lr=1e-3, swa 1e-4 20 => .39
- Orchid 78: 512/512, d=.1, wd=1e-3, lr=1e-4, swa 1e-5 20 =>
- Orchid 79: 512/512, d=.1, wd=1e-4, lr=1e-4, swa 1e-5 20 => .5899

DB LOSS

- Orchid 80: same => .79
- Orchid 81: same, preserve aspect ratio => .82
