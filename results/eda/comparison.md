# EDA comparison: `eda_person` vs `eda`

## Overview

|                       |       old |        new |
|:----------------------|----------:|-----------:|
| classes               |  1        |  70        |
| sequences             | 20        |   1.4e+03  |
| frames                |  6.41e+04 |   3.52e+06 |
| hours                 |  0.59     |  32.6      |
| % videos with FOC     | 45        |  37.3      |
| % videos with OOV     |  5        |  32        |
| % frames absent       |  0.301    |   2.79     |
| absence episodes      | 31        |   5.15e+03 |
| resolutions           |  9        |  56        |
| sequences with issues |  7        | 594        |

## Per-sequence statistics (mean)

|                |        old |       new |   new / old |
|:---------------|-----------:|----------:|------------:|
| n_frames       |   3.21e+03 |  2.51e+03 |       0.784 |
| duration_s     | 107        | 83.7      |       0.784 |
| frac_foc       |   0.00283  |  0.0126   |       4.45  |
| frac_oov       |   0.00237  |  0.0149   |       6.29  |
| frac_absent    |   0.0052   |  0.0275   |       5.29  |
| n_foc_episodes |   1.5      |  2.41     |       1.61  |
| n_oov_episodes |   0.05     |  1.27     |      25.4   |
| max_foc_len    |   3.6      | 15.2      |       4.24  |
| max_oov_len    |   2.5      | 18.3      |       7.33  |

## Absence-episode length (frames)

|                     |     count |   mean |    std |   min |   median |   p90 |        max |
|:--------------------|----------:|-------:|-------:|------:|---------:|------:|-----------:|
| ('old', 'full_occ') | 30        |   4.77 |   3.92 |     1 |        4 |  10.2 |  15        |
| ('old', 'out_view') |  1        |  50    | nan    |    50 |       50 |  50   |  50        |
| ('new', 'full_occ') |  3.37e+03 |  13.6  |  42.9  |     1 |        4 |  31   |   1.31e+03 |
| ('new', 'out_view') |  1.78e+03 |  29.2  |  52.8  |     1 |       12 |  69   | 565        |

## Attribute prevalence (% of videos)

Old run had derived proxies only; new run uses official labels where available.

|     |   old % |   new % |   new % (old classes) |
|:----|--------:|--------:|----------------------:|
| SV  |     100 |    94.4 |                   100 |
| ARC |      90 |    85.8 |                    90 |
| LR  |      35 |    47.2 |                    35 |
| FM  |      30 |    21.1 |                    15 |
| FOC |      45 |    38.7 |                    45 |
| OV  |       5 |    34.1 |                     5 |
| IV  |     nan |    16   |                    35 |
| POC |     nan |    65.2 |                    75 |
| DEF |     nan |    51.8 |                    85 |
| MB  |     nan |    32.1 |                    60 |
| CM  |     nan |    30.9 |                    45 |
| ROT |     nan |    58   |                    35 |
| BC  |     nan |    31   |                    80 |
| VC  |     nan |    10.9 |                     5 |

## Consistency on overlapping sequences

All 20 overlapping sequences match.

## Classes ranked by mean fraction of absent frames

| cls          |   videos |   mean_frames |   frac_absent |   pct_videos_foc |   pct_videos_oov |   rank_absent | in_old_run   |
|:-------------|---------:|--------------:|--------------:|-----------------:|-----------------:|--------------:|:-------------|
| mouse        |       20 |      2.06e+03 |       0.13    |               70 |               80 |             1 | False        |
| microphone   |       20 |      2.77e+03 |       0.129   |               75 |               55 |             2 | False        |
| volleyball   |       20 |      1.56e+03 |       0.113   |               70 |               90 |             3 | False        |
| fox          |       20 |      2.2e+03  |       0.0744  |               55 |               60 |             4 | False        |
| goldfish     |       20 |      3.02e+03 |       0.0693  |               60 |               50 |             5 | False        |
| gecko        |       20 |      2.41e+03 |       0.0681  |               40 |               60 |             6 | False        |
| basketball   |       20 |      2.38e+03 |       0.0671  |               80 |               50 |             7 | False        |
| helmet       |       20 |      2.78e+03 |       0.067   |               65 |               75 |             8 | False        |
| electricfan  |       20 |      1.97e+03 |       0.0657  |               40 |               65 |             9 | False        |
| yoyo         |       20 |      3.05e+03 |       0.0651  |               85 |               85 |            10 | False        |
| giraffe      |       20 |      2e+03    |       0.0634  |               30 |               35 |            11 | False        |
| monkey       |       20 |      2.42e+03 |       0.0587  |               50 |               45 |            12 | False        |
| hat          |       20 |      3.15e+03 |       0.0486  |               70 |               55 |            13 | False        |
| lizard       |       20 |      2.73e+03 |       0.0481  |               40 |               50 |            14 | False        |
| lion         |       20 |      2.51e+03 |       0.0469  |               45 |               45 |            15 | False        |
| gorilla      |       20 |      2.37e+03 |       0.0456  |               50 |               50 |            16 | False        |
| frog         |       20 |      1.95e+03 |       0.0434  |               45 |               65 |            17 | False        |
| licenseplate |       20 |      2.54e+03 |       0.0422  |               25 |               35 |            18 | False        |
| kite         |       20 |      3.15e+03 |       0.0366  |               15 |               65 |            19 | False        |
| elephant     |       20 |      2.12e+03 |       0.0351  |               40 |               25 |            20 | False        |
| hand         |       20 |      2.69e+03 |       0.0345  |               40 |               40 |            21 | False        |
| sheep        |       20 |      2.3e+03  |       0.0324  |               40 |               45 |            22 | False        |
| hippo        |       20 |      2.11e+03 |       0.0316  |               25 |               35 |            23 | False        |
| kangaroo     |       20 |      2.39e+03 |       0.028   |               50 |               35 |            24 | False        |
| coin         |       20 |      1.87e+03 |       0.0235  |               75 |               15 |            25 | False        |
| leopard      |       20 |      2.47e+03 |       0.0221  |               25 |               55 |            26 | False        |
| gametarget   |       20 |      2.31e+03 |       0.0216  |               55 |               40 |            27 | False        |
| pool         |       20 |      4.31e+03 |       0.0204  |               70 |               25 |            28 | False        |
| swing        |       20 |      2.31e+03 |       0.0204  |               50 |               35 |            29 | False        |
| drone        |       20 |      2.55e+03 |       0.0188  |               30 |               60 |            30 | False        |
| bicycle      |       20 |      2.03e+03 |       0.0182  |               40 |               25 |            31 | False        |
| motorcycle   |       20 |      2.2e+03  |       0.0182  |               60 |               40 |            32 | False        |
| bear         |       20 |      2.7e+03  |       0.0171  |               20 |               20 |            33 | False        |
| tank         |       20 |      2.79e+03 |       0.0167  |               55 |               10 |            34 | False        |
| zebra        |       20 |      2.26e+03 |       0.0162  |               40 |               25 |            35 | False        |
| dog          |       20 |      1.94e+03 |       0.0158  |               50 |               25 |            36 | False        |
| spider       |       20 |      3.2e+03  |       0.0155  |               10 |               60 |            37 | False        |
| cattle       |       20 |      1.84e+03 |       0.0144  |               25 |               15 |            38 | False        |
| surfboard    |       20 |      2.66e+03 |       0.0143  |               40 |               25 |            39 | False        |
| bottle       |       20 |      2.35e+03 |       0.0136  |               40 |               20 |            40 | False        |
| deer         |       20 |      2.4e+03  |       0.0135  |               15 |               20 |            41 | False        |
| pig          |       20 |      2.03e+03 |       0.0135  |               35 |               10 |            42 | False        |
| racing       |       20 |      2.62e+03 |       0.0133  |               45 |               25 |            43 | False        |
| truck        |       20 |      3.67e+03 |       0.0133  |               30 |               25 |            44 | False        |
| robot        |       20 |      3.81e+03 |       0.0126  |               40 |               30 |            45 | False        |
| turtle       |       20 |      3.25e+03 |       0.0113  |               40 |               20 |            46 | False        |
| guitar       |       20 |      2.3e+03  |       0.00955 |               30 |               20 |            47 | False        |
| train        |       20 |      2.34e+03 |       0.00724 |               30 |                5 |            48 | False        |
| rubicCube    |       20 |      2.29e+03 |       0.00654 |               30 |               10 |            49 | False        |
| rabbit       |       20 |      2.52e+03 |       0.00643 |               15 |               30 |            50 | False        |
| tiger        |       20 |      2.2e+03  |       0.00612 |               30 |               15 |            51 | False        |
| crocodile    |       20 |      1.89e+03 |       0.00607 |               30 |               10 |            52 | False        |
| shark        |       20 |      2.07e+03 |       0.00595 |               30 |               10 |            53 | False        |
| bus          |       20 |      3.08e+03 |       0.00549 |               10 |               15 |            54 | False        |
| person       |       20 |      3.21e+03 |       0.0052  |               45 |                5 |            55 | True         |
| flag         |       20 |      2.23e+03 |       0.00516 |               10 |               10 |            56 | False        |
| car          |       20 |      2.71e+03 |       0.00494 |               35 |                0 |            57 | False        |
| crab         |       20 |      1.86e+03 |       0.00479 |               25 |               20 |            58 | False        |
| cat          |       20 |      2.41e+03 |       0.00471 |               20 |               20 |            59 | False        |
| umbrella     |       20 |      3.56e+03 |       0.00462 |               35 |               25 |            60 | False        |
| skateboard   |       20 |      3.55e+03 |       0.00448 |               10 |               15 |            61 | False        |
| horse        |       20 |      2.75e+03 |       0.00445 |               15 |               20 |            62 | False        |
| squirrel     |       20 |      1.63e+03 |       0.0043  |               10 |               15 |            63 | False        |
| sepia        |       20 |      1.96e+03 |       0.00408 |                5 |               25 |            64 | False        |
| bird         |       20 |      2.43e+03 |       0.00368 |               25 |               20 |            65 | False        |
| book         |       20 |      2.08e+03 |       0.00348 |                5 |                0 |            66 | False        |
| cup          |       20 |      2.05e+03 |       0.00321 |               25 |                5 |            67 | False        |
| chameleon    |       20 |      1.74e+03 |       0.00253 |               15 |                5 |            68 | False        |
| boat         |       20 |      2.49e+03 |       0.00191 |               15 |               10 |            69 | False        |
| airplane     |       20 |      4.3e+03  |       0.0019  |               15 |               10 |            70 | False        |
