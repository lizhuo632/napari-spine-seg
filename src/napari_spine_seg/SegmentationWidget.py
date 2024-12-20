import numpy as np
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QGroupBox,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from napari_spine_seg.dl._2predict import predict


class SegmentationWidget(QWidget):
    def __init__(self, napari_viewer):
        super().__init__()
        self.viewer = napari_viewer

        self.initUI()
        self.viewer.events.layers_change.connect(self.update_layer_combo)

        # self.viewer.layers.events.changed.connect(self.update_layer_combo)

    def initUI(self):

        # 整个插件的布局
        layout = QVBoxLayout()
        self.setLayout(layout)

        # 第一部分Segmentation Group
        self.segmentation_group = QGroupBox("Segmentation")
        segmentation_layout = QVBoxLayout()
        self.segmentation_group.setLayout(segmentation_layout)
        layout.addWidget(self.segmentation_group)

        ## 从当前存在的layer中选择
        self.layer_label = QLabel("Layer:")
        segmentation_layout.addWidget(self.layer_label)
        self.layer_combo = QComboBox()
        self.update_layer_combo()
        segmentation_layout.addWidget(self.layer_combo)

        ## 选择类型
        self.type_label = QLabel("Image Type:")
        segmentation_layout.addWidget(self.type_label)
        self.type_combo = QComboBox()
        self.type_combo.addItem("actin")
        self.type_combo.addItem("pre-synapse")
        self.type_combo.addItem("psd")
        self.type_combo.addItem("myr")
        segmentation_layout.addWidget(self.type_combo)
        self.type_combo.currentIndexChanged.connect(self.update_target_combo)

        ## 分割目标
        self.target_label = QLabel("Segmentation Target:")
        segmentation_layout.addWidget(self.target_label)
        self.target_combo = QComboBox()
        self.update_target_combo()
        segmentation_layout.addWidget(self.target_combo)

        ## device
        self.device_label = QLabel("Device:")
        segmentation_layout.addWidget(self.device_label)
        self.device_line = QComboBox()
        self.device_line.addItem("cpu")
        self.device_line.addItem("cuda")

        ## 勾选normalize,鼠标悬停显示说明
        self.normalize_checkbox = QCheckBox("Normalize")
        self.normalize_checkbox.setToolTip(
            "patched images have been normalized already, so no need to normalize again"
        )
        segmentation_layout.addWidget(self.normalize_checkbox)

        ## 勾选仅处理当前帧
        self.current_frame_checkbox = QCheckBox("Only Process Current Frame")
        segmentation_layout.addWidget(self.current_frame_checkbox)

        ## Segmentation按钮
        self.segmentation_button = QPushButton("Segmentation")
        self.segmentation_button.clicked.connect(self.segment)
        segmentation_layout.addWidget(self.segmentation_button)

    def update_layer_combo(self, event=None):
        """
        根据 viewer.layers 更新 layer_combo 的内容
        """

        current_layer = self.layer_combo.currentText()
        # show_info("current_layer: " + current_layer)

        self.layer_combo.clear()
        for layer in self.viewer.layers:
            # show_info("layer: " + layer.name)
            self.layer_combo.addItem(layer.name)
        # 尝试恢复当前选择的 layer
        index = self.layer_combo.findText(current_layer)
        if index >= 0:
            self.layer_combo.setCurrentIndex(index)

    def update_target_combo(self):
        """
        根据选择的类型更新 target_combo 的内容
        """

        current_layer = self.layer_combo.currentText()
        type = self.type_combo.currentText()
        self.target_combo.clear()
        if type == "actin":
            self.target_combo.addItem("spine")
            self.target_combo.addItem("dendritre")
        elif type == "pre-synapse":
            self.target_combo.addItem("pre-synapse")
        elif type == "psd":
            self.target_combo.addItem("psd")
        elif type == "myr":
            self.target_combo.addItem("spine")
            self.target_combo.addItem("dendritre")

        index = self.target_combo.findText(current_layer)
        if index >= 0:
            self.target_combo.setCurrentIndex(index)

    def segment(self):
        # 获取layer
        layer_name = self.layer_combo.currentText()
        layer = self.viewer.layers[layer_name]
        data = layer.data
        type = self.type_combo.currentText()
        target = self.target_combo.currentText()
        device = self.device_line.currentText()
        normalize = self.normalize_checkbox.isChecked()
        current_frame = self.current_frame_checkbox.isChecked()
        if current_frame:
            frame = self.viewer.dims.current_step[0]
            data = data[frame]
        segmentation_result = predict(data, type, target, device, normalize)
        if current_frame:
            # 填充result到与layer尺寸相同
            result = np.zeros_like(layer.data)
            result[frame] = segmentation_result
            layer = self.viewer.add_labels(
                result, name="segmentation_result", scale=layer.scale
            )
        else:
            layer = self.viewer.add_labels(
                segmentation_result,
                name="segmentation_result",
                scale=layer.scale,
            )


# if __name__ == "__main__":
