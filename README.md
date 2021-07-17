# feature-selector
A custom feature selection program with various algorithms and their pipelined combination in an easy and descriptive way. 

 import FeatureSelector as fs
 X,y = fs.read_path('D:\Internship\dataset.csv')
 X,y
(       TSPAN6       FGR       CFH      GCLC    NIPAL3  ...    IFNGR1    SH2D2A  TNFRSF1B    ARNTL2      IBSP
0    3.593240  0.714444  3.119220  2.464426  1.291976  ...  4.462061  0.211963  3.097346  0.179946  0.024523
1    3.399946  0.654546  0.136111  2.003238  2.233912  ...  4.708507  0.100548  2.800072  0.076992  0.000000
2    3.469919  0.873279  0.815520  2.537203  1.894536  ...  3.753501  0.103287  2.079879  0.659337  0.084467
3    2.821973  0.648664  0.515149  2.874994  1.227373  ...  4.472364  1.024236  2.113983  0.131032  0.234907
4    2.113599  3.069690  1.088918  2.917765  1.872202  ...  5.070540  0.802297  5.009232  0.480860  4.219627
..        ...       ...       ...       ...       ...  ...       ...       ...       ...       ...       ...
495  2.035607  1.083142  1.969201  2.342382  2.938695  ...  4.184614  0.393621  3.387899  0.278416  1.163159
496  2.455688  2.287689  3.733934  2.194991  2.633475  ...  5.299762  2.237891  4.531835  1.525031  0.342196
497  3.077339  2.696516  3.127983  2.612161  2.994722  ...  5.178988  2.594080  4.550842  2.330173  0.931543
498  4.398039  2.100173  3.080552  3.052334  2.392759  ...  4.791044  0.457323  3.195818  0.842771  3.581545
499  2.129841  1.899062  0.582394  2.604003  2.158955  ...  4.414640  0.269112  2.584676  0.116136  1.627639

[500 rows x 198 columns], 0      0
1      0
2      1
3      0
4      1
      ..
495    1
496    0
497    1
498    0
499    0
Name: 0, Length: 500, dtype: int64)



>>> X,y = fs.read_path('D:\Internship\data.csv')
>>> X,y
(     mean_radius  mean_texture  mean_perimeter  mean_area  mean_smoothness
0          17.99         10.38          122.80     1001.0          0.11840
1          20.57         17.77          132.90     1326.0          0.08474
2          19.69         21.25          130.00     1203.0          0.10960
3          11.42         20.38           77.58      386.1          0.14250
4          20.29         14.34          135.10     1297.0          0.10030
..           ...           ...             ...        ...              ...
564        21.56         22.39          142.00     1479.0          0.11100
565        20.13         28.25          131.20     1261.0          0.09780
566        16.60         28.08          108.30      858.1          0.08455
567        20.60         29.33          140.10     1265.0          0.11780
568         7.76         24.54           47.92      181.0          0.05263

[569 rows x 5 columns], 0      0
1      0
2      0
3      0
4      0
      ..
564    0
565    0
566    0
567    0
568    1
Name: diagnosis, Length: 569, dtype: int64)


>>> X_feature_selected = fs.ForwardSelector(X,y)
>>> X_feature_selected
       HS3ST1   TSPOAP1      CROT     CRLF1  MAPK8IP2  PRICKLE3   SLC38A5
0    0.283871  1.118907  3.199585  0.602297  0.583563  2.728145  0.236678
1    0.778597  1.428139  1.861609  0.326153  0.860734  1.728958  0.138003
2    0.925642  0.928361  2.316242  0.061074  0.343178  2.808658  0.033121
3    2.460786  1.649740  2.610303  1.118434  1.612719  2.117057  0.101612
4    1.176990  0.769621  2.515964  0.164966  1.731133  2.934570  0.197540
..        ...       ...       ...       ...       ...       ...       ...
495  0.491308  0.722332  1.771241  0.525178  4.345678  2.425843  0.140930
496  0.749313  1.367277  2.886011  0.932400  2.602531  1.879879  1.344855
497  1.156488  1.327671  1.679746  0.672549  1.169465  2.129853  2.522259
498  0.622989  2.347412  4.090374  1.727858  2.867727  1.990572  0.845473
499  0.311338  2.481399  3.411503  0.381369  3.573363  2.402876  1.390478

[500 rows x 7 columns]


>>> X_feature_selected = fs.Pipelined(X,y)
>>> X_feature_selected
         CROT   SLC38A5
0    3.199585  0.236678
1    1.861609  0.138003
2    2.316242  0.033121
3    2.610303  0.101612
4    2.515964  0.197540
..        ...       ...
495  1.771241  0.140930
496  2.886011  1.344855
497  1.679746  2.522259
498  4.090374  0.845473
499  3.411503  1.390478

[500 rows x 2 columns]
