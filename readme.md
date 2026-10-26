# 基于单目视觉的目标物测量装置视觉部分

![1](/docs/1.jpg)

[C题_基于单目视觉的目标物测量装置](/docs/C题_基于单目视觉的目标物测量装置.pdf)

## 硬件部分

- 树莓派4B
- 海康工业相机MV-CA013-21UC
- 7寸触摸屏

## 软件部分

- Ubuntu22.04
- Python>=3.9
- opencv-python>=4.5
- PyQT>=5.15
- numpy>=1.21
- HikRobot MVS SDK

## 文件结构

```text
core/
  main.py          CLI 和 GUI 入口
  Camera.py        海康 SDK / OpenCV 相机 / 图片输入
  SolverPnP.py     平面位姿、相机光心到目标参考点的距离
  Base.py          基础图形识别与测量
  Advanced.py      正方形候选融合、离线数字识别
  advanced_geometry.py  数字孔洞处理、局部边重建与鲁棒几何拟合
  vision.py        A4 边框检测、去畸变、透视校正、结果结构
  config.py        配置检查和已有标定结果写入接口
  session.py       稳定性检查及记录保存
  QTUI.py          界面逻辑和采集/测量线程
  DebugPanel.py    适应性调试参数控件
  tuning.py        调参白名单、范围、默认值和校验
  QTUI.ui          Qt Designer 界面布局
  common.json      内参、相机、纸张及算法参数
AutoStart.sh       窗口模式启动脚本
register_service.sh  系统级 systemd 服务一键注册
```

## 安装与启动

在图形桌面的终端里启动：

```bash
cd /home/cwb/vision_code
bash AutoStart.sh
```

### systemd 自启动服务
- 注册服务

```bash
# 进入当前项目后输入：
chmod +x ./register_service.sh
sudo ./register_service.sh

# 正常时有如下输出
# Creating systemd service file at /etc/systemd/system/vision-measurement.service...
# Reloading systemd daemon...
# Enabling service vision-measurement.service...
# Starting service vision-measurement.service...
# Service vision-measurement.service has been registered and started.
```
- 查看程序状态

```bash
systemctl status vision-measurement
```
- 查看终端输出

```bash
sudo journalctl -u vision-measurement -b -f
```
- 关闭程序

```bash
sudo systemctl stop vision-measurement
```
- 重启程序

```bash
sudo systemctl restart vision-measurement
```
- 取消自启动（不会停止当前进程；如需立即关闭，再执行停止命令）

```bash
sudo systemctl disable vision-measurement
```

- 恢复自启动并启动程序

```bash
sudo systemctl enable --now vision-measurement
```
## 适应性调试

在 UI 顶部勾选 **适应性调试**，调试区五个数值控件：

- 曝光时间（µs）、增益：通过海康 SDK 在线设置，并回读设备实际值及允许范围。
- 固定二值化阈值：0～255。
- Canny 低阈值、高阈值：0～255。

原图、二值图、Canny 边缘和四边形候选预览仍可切换。四边形候选是筛选提示。

**保存参数**：将这五个参数合并到当前配置文件。
**恢复进入前**: 重新应用进入本次调试时的数值。

## 测量结果
1.基础题
- 正方形：测量：D:169.72cm x：11.90cm  实际：D:172.4cm x：12cm
- 等边三角形：测量：D:148.46cm x：13.03cm  实际：D:151.7cm x：13cm 
- 圆形：测量：D:131.91cm x：10.41cm  实际：D:135.2cm x：10.5cm
  
2.发挥题
- (1) 测量：D:151.41cm 最小正方形x：7.14cm  实际：D：154.3cm 最小正方形x：7cm
- (2) 测量：D:167.56cm 最小正方形x：6.28cm  实际：D：172.8cm 最小正方形x：6cm
- (3) 测量：D:135.16cm 指定数字的正方形x：9.37cm  实际：D：140.1cm 指定数字的正方形x：10cm
- (4) 测量：正方形x：6.87cm  实际：正方形x：8cm