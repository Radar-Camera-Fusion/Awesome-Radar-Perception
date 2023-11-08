# <img src='image/logo.png' height=30 /> Awesome Radar Perception 

## Overview


- [ Awesome Radar Perception](#-awesome-radar-perception)
  - [Overview](#overview)
  - [Surveys](#surveys)
  - [Datasets](#datasets)
  - [Representations](#representations)
    - [ADC Signal](#adc-signal)
      - [Classification/Motion Recognition](#classificationmotion-recognition)
      - [Object Dection](#object-dection)
    - [Radar Tensor](#radar-tensor)
      - [Detection](#detection)
      - [Segmentation](#segmentation)
      - [Multi-Task](#multi-task)
    - [Point Cloud](#point-cloud)
      - [Classification](#classification)
      - [Detection](#detection-1)
      - [Segmentation](#segmentation-1)
      - [Tracking](#tracking)
      - [Odometry](#odometry)
      - [Gait Recognition](#gait-recognition)
    - [Grid Map](#grid-map)
      - [Detection](#detection-2)
      - [Segmentation](#segmentation-2)
    - [Micro-Doppler Signature](#micro-doppler-signature)
      - [Motion (Gait/Gestures/Activity) Classification](#motion-gaitgesturesactivity-classification)
  - [Calibration](#calibration)
  - [Citation](#citation)


## Surveys
* 2023 - Reviewing 3D Object Detectors in the Context of High-Resolution 3+1D Radar __`CVPRW`__ [[Paper](https://arxiv.org/abs/2308.05478)]
* 2023 - Radar-Camera Fusion for Object Detection and Semantic Segmentation in Autonomous Driving: A Comprehensive Review __`TIV`__ [[Paper](https://ieeexplore.ieee.org/document/10225711)] [[Website](https://radar-camera-fusion.github.io)] [[GitHub](https://github.com/Radar-Camera-Fusion/Awesome-Radar-Camera-Fusion)]
* 2023 - Radars for Autonomous Driving: A Review of Deep Learning Methods and Challenges __`ACCESS`__ [[Paper](https://ieeexplore.ieee.org/document/10242101)]
* 2023 - Vehicle Detection for Autonomous Driving: A Review of Algorithms and Datasets __`TITS`__ [[Paper](https://ieeexplore.ieee.org/abstract/document/10196355)]
  
## Datasets
| Id | Name             | Year | Task                                                                                                      | Annotation                     | Data Representation                                   | Link                                                         |
|----|------------------|------|-----------------------------------------------------------------------------------------------------------|--------------------------------|-------------------------------------------------------------|--------------------------------------------------------------|
| 1  | nuScenes         | 2019 | Object Detection<br>Object Tracking                                                                       | 3D box                   | Point Cloud                                                 | [Paper](https://www.nuscenes.org/nuscenes) [Website](https://www.nuscenes.org/nuscenes) [Github](https://github.com/nutonomy/nuscenes-devkit)                           |
| 2  | Astyx            | 2019 | Object Detection                                                                                          | 3D box                   | Point Cloud                                                 | [Website](http://www.astyx.net)                                         |
| 3  | SeeingThroughFog | 2020 | Object Detection                                                                                          | 2D box<br>3D box   | Point Cloud                                                 | [Website](https://www.uni-ulm.de/en/in/driveu/projects/dense-datasets/) |
| 4  | CARRADA          | 2020 | Object Detection<br>Semantic Segmentation<br>Object Tracking<br>Trajectory Prediction                     | 2D box<br>2D pixel | Range-Doppler Tensor<br>Range-Azimuth Tensor                | [Website](https://arthurouaknine.github.io/codeanddata/carrada)                                                             |
| 5  | HawkEye          | 2020 | Semantic Segmentation                                                                                     | 3D point                 | Point Cloud                                                 | [Website](https://jguan.page/HawkEye/)                                                             |
| 6  | Zendar           | 2020 | Object Detection<br>Mapping<br>Localization                                                               | 2D box                   | Range-Doppler Tensor<br>Range-Azimuth Tensor<br>Point Cloud | [Website](http://zendar.io/dataset)                                                          |
| 7  | RADIATE          | 2020 | Object Detection<br>Object Tracking<br>SLAM<br>Scene Understanding                                        | 2D box                   | Range-Azimuth Tensor                                        | [Website](http://pro.hw.ac.uk/radiate/)                                                            |
| 8  | AIODrive         | 2020 | Object Detection<br>Object Tracking<br>Semantic Segmentation<br>Trajectory Prediction<br>Depth Estimation | 2D box<br>3D box   | Point Cloud                                                 | [Website](http://www.aiodrive.org/)                                                            |
| 9  | CRUW             | 2021 | Object Detection                                                                                          | 2D box                   | Range-Azimuth Tensor                                        | [Website](https:/www.cruwdataset.org/)                                                             |
| 10 | RaDICaL          | 2021 | Object Detection                                                                                          | 2D box                   | ADC Signal                                                  | [Website](https://publish.illinois.edu/radicaldata/)                                                             |
| 11 | RadarScenes | 2021 | Object Detection<br>Semantic Segmentation <br>Object Tracking   | 2D pixel<br>3D point | Point Cloud                                                                                               |  [Website](https://radar-scenes.com/) |
| 12 | RADDet      | 2021 | Object Detection                             | 2D box<br>3D box     | Range-Azimuth-Doppler Tensor                                                                                              |  [Github](https://github.com/ZhangAoCanada/RADDet) |
| 13 | FloW        | 2021 | Object Detection                             | 2D box                     | Range-Doppler Tensor<br>Point Cloud                                                                                               | [Website](https://orca-tech.cn/datasets/FloW/Introduction) [Github](https://github.com/ORCA-Uboat/FloW-Dataset)  |
| 14 | RADIal      | 2021 | Object Detection<br>Semantic Segmentation    | 2D box                     | ADC Signal<br>Range-Azimuth-Doppler Tensor<br>Range-Azimuth Tensor<br>Range-Doppler Tensor<br>Point Cloud             |  [Github](https://github.com/valeoai/RADIal) |
| 15 | VoD         | 2022 | Object Detection                             | 2D box<br>3D box     | __`4D`__ Point Cloud                                                                                                         | [Website](https://tudelft-iv.github.io/view-of-delft-dataset/)  |
| 16 | Boreas      | 2022 | Object Detection<br>Localization<br>Odometry | 2D box                     | Range-Azimuth Tensor                                                                                                  |  [Website](https://www.boreas.utias.utoronto.ca/) |
| 17 | TJ4DRadSet  | 2022 | Object Detection<br>Object Tracking          | 3D box                     | __`4D`__ Point Cloud                                                                                                               | [Website](https://github.com/TJRadarLab/TJ4DRadSet)  |
| 18 | K-Radar     | 2022 | Object Detection<br>Object Tracking<br>SLAM  | 3D box                     | __`4D`__ Range-Azimuth-Doppler Tensor                                                                                                         | [Github](https://github.com/kaist-avelab/k-radar)  |
| 19 | aiMotive    | 2022 | Object Detection                             | 3D box                     | Point cloud                                                                                                                       |  [Website](https://github.com/aimotive/aimotive_dataset) |
| 20 | WaterScenes    | 2023 | Instance Segmentation<br>Semantic Segmentation<br>Free-space Segmentation<br>Waterline Segmentation<br>Panoptic Perception          | 2D box<br>2D pixel<br>2D line<br>3D point               | __`4D`__ Point cloud                                                           |  [Paper](https://arxiv.org/abs/2307.06505) [Website](https://waterscenes.github.io) [GitHub](https://github.com/waterscenes/waterscenes) |
| 21 | ThermRad | 2023 | Object Detection | 3D box | __`4D`__ Point Cloud | [Paper](https://arxiv.org/abs/2308.10161)|
| 22 | Dual Radar | 2023 | Object Detection<br>Object Tracking | 3D box | __`4D`__ Point Cloud | [Paper](https://arxiv.org/abs/2310.0760) [GitHub](https://github.com/adept-thu/Dual-Radar)|


## Representations

### ADC Signal
#### Classification/Motion Recognition
* 2020 - Radar Image Reconstruction from Raw ADC Data using Parametric Variational Autoencoder with Domain Adaptation __`ICPR`__ [[Paper](https://ieeexplore.ieee.org/abstract/document/9412858)]
* 2021 - Improved Target Detection and Feature Extraction using a Complex-Valued Adaptive Sine Filter on Radar Time Domain Data __`EUSIPCO`__ [[Paper](https://ieeexplore.ieee.org/abstract/document/9616250)]
* 2021 - Data-Driven Radar Processing Using a Parametric Convolutional Neural Network for Human Activity Classification __`IEEE Sensors`__ [[Paper](https://ieeexplore.ieee.org/abstract/document/9464267)]
* 2023 - CubeLearn: End-to-End Learning for Human Motion Recognition From Raw mmWave Radar Signals __`IEEE IOT`__ [[Paper](https://ieeexplore.ieee.org/abstract/document/10018429)]

#### Object Dection
* 2023 - T-FFTRadNet: Object Detection with Swin Vision Transformers from Raw ADC Radar Signals __`arXiv`__ [[Paper](https://arxiv.org/abs/2303.16940)]
* 2023 - Echoes Beyond Points: Unleashing the Power of Raw Radar Data in Multi-modality Fusion __`arXiv`__ [[Paper](https://arxiv.org/abs/2307.16532)]


### Radar Tensor
#### Detection
* 2019 - Experiments with mmWave Automotive Radar Test-bed __`RA`__ __`ACSSC`__ [[Paper](https://ieeexplore.ieee.org/abstract/document/9048939)]
* 2019 - Vehicle Detection With Automotive Radar Using Deep Learning on Range-Azimuth-Doppler Tensors __`RAD`__ __`ICCVW`__ [[Paper](https://ieeexplore.ieee.org/document/9022248)]
* 2020 - Probabilistic oriented object detection in automotive radar __`RA`__ __`CVPRW`__ [[Paper](https://ieeexplore.ieee.org/document/9150751)]
* 2020 - RODNet: Radar Object Detection Using Cross-Modal Supervision __`RA`__  __`WACV`__ [__`CRUW`__] [[Paper](https://openaccess.thecvf.com/content/WACV2021/papers/Wang_RODNet_Radar_Object_Detection_Using_Cross-Modal_Supervision_WACV_2021_paper.pdf)]
* 2020 - RODNet: A Real-Time Radar Object Detection Network Cross-Supervised by Camera-Radar Fused Object 3D Localization __`RA`__ __`JSTSP`__  [__`CRUW`__] [[Paper](https://ieeexplore.ieee.org/document/9353210)]
* 2020 - Range-Doppler Detection in Automotive Radar with Deep Learning __`RD`__ __`IJCNN`__ [[Paper](https://ieeexplore.ieee.org/document/9207080)]
* 2020 - RAMP-CNN: A Novel Neural Network for Enhanced Automotive Radar Object Recognition __`RAD`__ __`IEEE Sensors`__ [[Paper](https://ieeexplore.ieee.org/abstract/document/9249018)]
* 2020 - CNN Based Road User Detection Using the 3D Radar Cube __`RAD`__ __`RAL`__ [[Paper](https://ieeexplore.ieee.org/abstract/document/8962258)]
* 2021 - [**GTR-Net**] Graph Convolutional Networks for 3D Object Detection on Radar Data __`RAD`__ __`ICCV Workshop`__ [[Paper](https://openaccess.thecvf.com/content/ICCV2021W/AVVision/html/Meyer_Graph_Convolutional_Networks_for_3D_Object_Detection_on_Radar_Data_ICCVW_2021_paper.html?ref=https://githubhelp.com)]
* 2021 - RADDet: Range-Azimuth-Doppler based Radar Object Detection for Dynamic Road Users __`RAD`__ __`CRV`__ [__`RADDet`__][[Paper](https://ieeexplore.ieee.org/document/9469418)] [[Code](https://github.com/ZhangAoCanada/RADDet)]
* 2022 - DAROD: A Deep Automotive Radar Object Detector on Range-Doppler maps __`RD`__ __`IV`__ [__`CARRADA`__ __`RADDet`__] [[Paper](https://ieeexplore.ieee.org/document/9827281)]
* 2022 - K-Radar: 4D Radar Object Detection for Autonomous Driving in Various Weather Conditions __`RADE`__ __`NeurIPS`__ [__`K-Radar`__][[Paper](https://proceedings.neurips.cc/paper_files/paper/2022/hash/185fdf627eaae2abab36205dcd19b817-Abstract-Datasets_and_Benchmarks.html)] [[GitHub](https://github.com/kaist-avelab/k-radar)]
* 2023 - Enhanced K-Radar: Optimal Density Reduction to Improve Detection Performance and Accessibility of 4D Radar Tensor-based Object Detection __`RADE`__ __`arXiv`__ [__`K-Radar`__][[Paper](https://arxiv.org/abs/2303.06342)]


#### Segmentation
* 2020 - RSS-Net: Weakly-supervised multi-class semantic segmentation with FMCW radar __`RAD`__ __`IV`__ [[Paper](https://ieeexplore.ieee.org/abstract/document/9304674)]
* 2020 - Deep Open Space Segmentation using Automotive Radar __`RAD`__ __`ICMIM`__ [[Paper](https://ieeexplore.ieee.org/abstract/document/9299052)]
* 2021 - PolarNet: Accelerated Deep Open Space Segmentation using Automotive Radar in Polar Domain  __`RAD`__ __`VEHITS`__ [[Paper](https://arxiv.org/abs/2103.03387)]
* 2021 - Multi-view Radar Semantic Segmentation __`RAD`__ __`ICCV`__ [[Paper](https://openaccess.thecvf.com/content/ICCV2021/html/Ouaknine_Multi-View_Radar_Semantic_Segmentation_ICCV_2021_paper.html)] [[GitHub](https://github.com/valeoai/MVRSS)]

#### Multi-Task
* 2022 - [**FFT-RadNet**] Raw High-Definition Radar for Multi-Task Learning __`CVPR`__ [[Paper](https://openaccess.thecvf.com/content/CVPR2022/html/Rebut_Raw_High-Definition_Radar_for_Multi-Task_Learning_CVPR_2022_paper.html)]
* 2023 - Cross-Modal Supervision-Based Multitask Learning With Automotive Radar Raw Data __`TIV`__ [[Paper](https://ieeexplore.ieee.org/abstract/document/10008067)]


### Point Cloud
#### Classification
* 2017 - Comparison of random forest and long short-term memory network performances in classification tasks using radar __`SDF`__ [[Paper](https://ieeexplore.ieee.org/abstract/document/8126350)]
* 2018 - Radar-based Feature Design and Multiclass Classification for Road User Recognition __`IV`__ [[Paper](https://ieeexplore.ieee.org/abstract/document/8500607)]
* 2020 - Off-the-shelf sensor vs. experimental radar - How much resolution is necessary in automotive radar classification? __`FUSION`__ [[Paper](https://ieeexplore.ieee.org/abstract/document/9190338)]
* 2022 - Radar-PointGNN: Graph Based Object Recognition for Unstructured Radar Point-cloud Data __`RadarConf`__ [[Paper](https://ieeexplore.ieee.org/abstract/document/9455172)]


#### Detection
* 2019 - 2D Car Detection in Radar Data with PointNets __`ITSC`__ [[Paper](https://ieeexplore.ieee.org/abstract/document/8917000)]
* 2020 - Detection and Tracking on Automotive Radar Data with Deep Learning __`FUSION`__ [[Paper](https://ieeexplore.ieee.org/abstract/document/9190261)]
* 2020 - Seeing Around Street Corners: Non-Line-of-Sight Detection and Tracking In-the-Wild Using Doppler Radar __`CVPR`__ [[Paper](https://openaccess.thecvf.com/content_CVPR_2020/html/Scheiner_Seeing_Around_Street_Corners_Non-Line-of-Sight_Detection_and_Tracking_In-the-Wild_Using_CVPR_2020_paper.html)]
* 2021 - RPFA-Net: a 4D RaDAR Pillar Feature Attention Network for 3D Object Detection __`ITSC`__ [[Paper](https://ieeexplore.ieee.org/abstract/document/9564754)] [[Code](https://github.com/adept-thu/RPFA-Net)]
* 2022 - Contrastive Learning for Automotive mmWave Radar Detection Points Based Instance Segmentation __`ITSC`__ [[Paper](https://ieeexplore.ieee.org/document/9922540)]
* 2023 - 3-D Object Detection for Multiframe 4-D Automotive Millimeter-Wave Radar Point Cloud __`IEEE Sensors`__ [__`TJ4DRadSet`__][[Paper](https://ieeexplore.ieee.org/abstract/document/9944629)]
* 2023 - SMURF: Spatial Multi-Representation Fusion for 3D Object Detection with 4D Imaging Radar __`TIV`__ [__`VoD`__ __`TJ4DRadSet`__][[Paper](https://ieeexplore.ieee.org/abstract/document/10274127)]
* 2023 - MVFAN: Multi-View Feature Assisted Network for 4D Radar Object Detection __`ICONIP`__ [__`Astyx`__ __`VoD`__][[Paper](https://arxiv.org/abs/2310.16389)]

#### Segmentation
* 2018 - Semantic Segmentation on Radar Point Clouds __`FUSION`__ [[Paper](https://ieeexplore.ieee.org/document/8455344)]
* 2018 - Supervised Clustering for Radar Applications: On the Way to Radar Instance Segmentation __`ICMIM`__ [[Paper](https://ieeexplore.ieee.org/abstract/document/8917000)]
* 2019 - 2D Car Detection in Radar Data with PointNets __`ITSC`__ [[Paper](https://ieeexplore.ieee.org/abstract/document/8917000)]
* 2020 - RSS-Net: Weakly-Supervised Multi-Class Semantic Segmentation with FMCW Radar __`IV`__ [[Paper](https://ieeexplore.ieee.org/abstract/document/9304674)]
* 2022 - Panoptic Segmentation for Automotive Radar Point Cloud __`RadarConf`__ [[Paper](https://ieeexplore.ieee.org/document/9764218)]
* 2023 - Deep Instance Segmentation With Automotive Radar Detection Points __`TIV`__ [__`RadarScenes`__] [[Paper](https://ieeexplore.ieee.org/abstract/document/9762032)]

<!-- Point cloud segmentation with a high-resolution automotive radar __``__ [[Paper](https://ieeexplore.ieee.org/document/8727840)] -->
<!-- Kernel point convolution LSTM networks for radar point cloud segmentation __`Applied Science`__ -->

#### Tracking
* 2020 - Detection and Tracking on Automotive Radar Data with Deep Learning __`FUSION`__ [[Paper](https://ieeexplore.ieee.org/abstract/document/9190261)]
* 2023 - Which Framework is Suitable for Online 3D Multi-Object Tracking for Autonomous Driving with Automotive 4D Imaging Radar? __`arXiv`__ [[Paper](https://arxiv.org/abs/2309.06036)]


#### Odometry
* 2023 - Efficient Deep-Learning 4D Automotive Radar Odometry Method __`TIV`__ [[Paper](https://ieeexplore.ieee.org/abstract/document/10237296)]
* 2023 - DRIO: Robust Radar-Inertial Odometry in Dynamic Environments __`RAL`__ [[Paper](https://ieeexplore.ieee.org/abstract/document/10207713)]

#### Gait Recognition
* 2021 - Person Reidentification Based on Automotive Radar Point Clouds __`TGRS`__ [[Paper](https://ieeexplore.ieee.org/document/9420713)]
* 2020 - Gait Recognition for Co-Existing Multiple People Using Millimeter Wave Sensing __`AAAI`__ [[Paper](https://ojs.aaai.org/index.php/AAAI/article/view/5430)]

### Grid Map
#### Detection
* 2015 - Automotive Radar Gridmap Representations __`ICMIM`__ [[Paper](https://ieeexplore.ieee.org/abstract/document/7117922)]
* 2015 - Detection of Arbitrarily Rotated Parked Cars Based on Radar Sensors __`IRS`__ [[Paper](https://ieeexplore.ieee.org/abstract/document/7226281)]
* 2016 - 3D Occupancy Grid Mapping Using Statistical Radar Models __`IV`__ [[Paper](https://ieeexplore.ieee.org/abstract/document/7535495)]
* 2017 - Semantic Radar Grids __`IV`__ [[Paper](https://ieeexplore.ieee.org/abstract/document/7995871)]
* 2018 - Adaptions for Automotive Radar Based Occupancy Gridmaps __`ICMIM`__ [[Paper](https://ieeexplore.ieee.org/abstract/document/8443484)]
* 2018 - High Resolution Radar-based Occupancy Grid Mapping and Free Space Detection __`VEHITS`__ [[Paper](https://pdfs.semanticscholar.org/d888/6334e15acebe688f993f45da7ba7bde79eff.pdf)]
* 2019 - Semantic Segmentation on Automotive Radar Maps __`IV`__ [[Paper](https://ieeexplore.ieee.org/abstract/document/8813808)]
* 2019 - Occupancy Grids Generation Using Deep Radar Network for Autonomous Driving __`ITSC`__ [[Paper](https://ieeexplore.ieee.org/abstract/document/8916897)]
* 2020 - Semantic Segmentation on 3D Occupancy Grids for Automotive Radar __`IEEE ACCESS`__ [[Paper](https://ieeexplore.ieee.org/abstract/document/9229096)]


#### Segmentation
* 2019 - Road Scene Understanding by Occupancy Grid Learning from Sparse Radar Clusters using Semantic Segmentation __`ICCV`__ [[Paper](https://openaccess.thecvf.com/content_ICCVW_2019/html/CVRSUAD/Sless_Road_Scene_Understanding_by_Occupancy_Grid_Learning_from_Sparse_Radar_ICCVW_2019_paper.html)]
* 2020 - Scene Understanding With Automotive Radar __`TIV`__ [[Paper](https://ieeexplore.ieee.org/abstract/document/8911477)]
* 2023 - Semantic Segmentation-Based Occupancy Grid Map Learning With Automotive Radar Raw Data __`TIV`__ [__`RADIal`__][[Paper](https://ieeexplore.ieee.org/abstract/document/10273590)]

### Micro-Doppler Signature
#### Motion (Gait/Gestures/Activity) Classification
* 2016 - Human Detection and Activity Classification Based on Micro-Doppler Signatures Using Deep Convolutional Neural Networks __`IEEE Geoscience and Remote Sensing Letters`__ [[Paper](https://ieeexplore.ieee.org/abstract/document/7314905)]
* 2017 - New Analysis of Radar Micro-Doppler Gait Signatures for Rehabilitation and Assisted Living __`ICASSP`__ [[Paper](https://ieeexplore.ieee.org/abstract/document/7952908)]
* 2018 - Human Motion Classification with Micro-Doppler Radar and Bayesian-Optimized Convolutional Neural Networks __`ICASSP`__ [[Paper](https://ieeexplore.ieee.org/abstract/document/8461847)]
* 2018 - Radar-Based Human-Motion Recognition With Deep Learning: Promising Applications for Indoor Monitoring __`IEEE Signal Processing Magazine`__ [[Paper](https://ieeexplore.ieee.org/abstract/document/8746862)]
* 2019 - Radar-Based Human Gait Recognition Using Dual-Channel Deep Convolutional Neural Network __`TGRS`__ [[Paper](https://ieeexplore.ieee.org/abstract/document/8789686)]
* 2019 - Experiments with mmWave Automotive Radar Test-bed __`ACSSC`__ [[Paper](https://ieeexplore.ieee.org/abstract/document/9048939)]
* 2022 - Attention-Based Dual-Stream Vision Transformer for Radar Gait Recognition __`ICASSP`__ [[Paper](https://ieeexplore.ieee.org/abstract/document/9746565)]
 <!-- Doppler-radar based hand gesture recognition system using convolutional neural networks -->
 <!-- Practical classification of different moving targets using automotive radar and deep neural networks -->

## Calibration
* 2019 - Extrinsic 6DoF calibration of a radar–LiDAR–camera system enhanced by radar cross section estimates evaluation [[Paper](https://www.sciencedirect.com/science/article/pii/S0921889018301994)]
* 2021 - A Joint Extrinsic Calibration Tool for Radar, Camera and Lidar __`TIV`__ [[Paper](https://ieeexplore.ieee.org/abstract/document/9380784)]
* 2021 - 3D Detection and Tracking for On-road Vehicles with a Monovision Camera and Dual Low-cost 4D mmWave Radars __`ITSC`__ [[Paper](https://ieeexplore.ieee.org/abstract/document/9564904)]
* 2022 - 3DRadar2ThermalCalib: Accurate Extrinsic Calibration between a 3D mmWave Radar and a Thermal Camera Using a Spherical-Trihedral __`ITSC`__ [[Paper](https://ieeexplore.ieee.org/abstract/document/9922522)]
* 2022 - K-Radar: 4D Radar Object Detection for Autonomous Driving in Various Weather Conditions  __`NeurIPS`__ [[Paper](https://proceedings.neurips.cc/paper_files/paper/2022/hash/185fdf627eaae2abab36205dcd19b817-Abstract-Datasets_and_Benchmarks.html)] [[GitHub](https://github.com/kaist-avelab/k-radar)]

## Citation
Please use the following citation when referencing
```
@article{yao2023radar,
  author={Yao, Shanliang and Guan, Runwei and Huang, Xiaoyu and Li, Zhuoxiao and Sha, Xiangyu and Yue, Yong and Lim, Eng Gee and Seo, Hyungjoon and Man, Ka Lok and Zhu, Xiaohui and Yue, Yutao},
  journal={IEEE Transactions on Intelligent Vehicles}, 
  title={Radar-Camera Fusion for Object Detection and Semantic Segmentation in Autonomous Driving: A Comprehensive Review}, 
  year={2023},
  volume={},
  number={},
  pages={1-40},
  doi={10.1109/TIV.2023.3307157}}
```
