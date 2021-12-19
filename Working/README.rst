.. raw:: html


**Important note**

    Một số kiểu máy của phiên bản '' 1. * '' không tương thích với các kiểu máy đã được đào tạo trước đó, 17 nếu bạn có các kiểu máy như vậy và muốn tải chúng - hãy quay lại với

    $ pip install -U segmentation-models==0.2.1

Table of Contents
~~~~~~~~~~~~~~~~~
 - `Quick start`_
 - `Simple training pipeline`_
 - `Examples`_
 - `Models and Backbones`_
 - `Installation`_
 - `Documentation`_
 
Quick start
~~~~~~~~~~~
Library được xây dựng để làm việc cùng với Keras và TensorFlow Keras frameworks

.. code:: python

    import segmentation_models as sm
    # Segmentation Models: using `keras` framework.

Mặc định là import ``keras``, nếu keras không được cài đặt trước đó, nó sẽ thay thế bằng ``tensorflow.keras`` framework.
rk:Có một vài cách để chọn framework:

- Cung cấp biến môi trường ``SM_FRAMEWORK=keras`` / ``SM_FRAMEWORK=tf.keras`` trước import ``segmentation_models``
- Đổi framework ``sm.set_framework('keras')`` /  ``sm.set_framework('tf.keras')``

.. code:: python

    import keras
    # or from tensorflow import keras

    keras.backend.set_image_data_format('channels_last')
    # or keras.backend.set_image_data_format('channels_first')
   
Tạo segmentation model chỉ là một ví dụ của Keras Model

.. code:: python
    
    model = sm.Unet()

Dựa trên nhiệm vụ, ta có thể chuyển đổi kiến trúc mạng bằng cách chọn các backbone với các tham số và pretrained weights đã được khởi tạo kèm theo:
.. code:: python

    model = sm.Unet('resnet34', encoder_weights='imagenet')


Thay đổi số output classes trong model:

.. code:: python
    
    # binary segmentation (this parameters are default when you call Unet('resnet34')
    model = sm.Unet('resnet34', classes=1, activation='sigmoid')
    
.. code:: python
    
    # multiclass segmentation with non overlapping class masks (your classes + background)
    model = sm.Unet('resnet34', classes=3, activation='softmax')
    
.. code:: python
    
    # multiclass segmentation with independent overlapping/non-overlapping class masks
    model = sm.Unet('resnet34', classes=3, activation='sigmoid')
    
    
Thay đổi input shape của model:

.. code:: python
    
    # if you set input channels not equal to 3, you have to set encoder_weights=None
    # how to handle such case with encoder_weights='imagenet' described in docs
    model = Unet('resnet34', input_shape=(None, None, 6), encoder_weights=None)
   
Simple training pipeline
~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

    import segmentation_models as sm

    BACKBONE = 'resnet34'
    preprocess_input = sm.get_preprocessing(BACKBONE)

    # load your data
    x_train, y_train, x_val, y_val = load_data(...)

    # preprocess input
    x_train = preprocess_input(x_train)
    x_val = preprocess_input(x_val)

    # define model
    model = sm.Unet(BACKBONE, encoder_weights='imagenet')
    model.compile(
        'Adam',
        loss=sm.losses.bce_jaccard_loss,
        metrics=[sm.metrics.iou_score],
    )

    # fit model
    # if you use data generator use model.fit_generator(...) instead of model.fit(...)
    # more about `fit_generator` here: https://keras.io/models/sequential/#fit_generator
    model.fit(
       x=x_train,
       y=y_train,
       batch_size=16,
       epochs=100,
       validation_data=(x_val, y_val),
    )

Các thao tác tương tự có thể được thực hiện với ``Linknet``, ``PSPNet`` and ``FPN``. Để biết thêm thông tin chi tiết về API mô hình và các trường hợp sử dụng `Read the Docs <https://segmentation-models.readthedocs.io/en/latest/>`__.


Models and Backbones
~~~~~~~~~~~~~~~~~~~~
**Models**

-  `Unet <https://arxiv.org/abs/1505.04597>`__
-  `FPN <http://presentations.cocodataset.org/COCO17-Stuff-FAIR.pdf>`__
-  `Linknet <https://arxiv.org/abs/1707.03718>`__
-  `PSPNet <https://arxiv.org/abs/1612.01105>`__

============= ==============
Unet          Linknet
============= ==============
|unet_image|  |linknet_image|
============= ==============

============= ==============
PSPNet        FPN
============= ==============
|psp_image|   |fpn_image|
============= ==============

.. _Unet: https://github.com/qubvel/segmentation_models/blob/readme/LICENSE
.. _Linknet: https://arxiv.org/abs/1707.03718
.. _PSPNet: https://arxiv.org/abs/1612.01105
.. _FPN: http://presentations.cocodataset.org/COCO17-Stuff-FAIR.pdf

.. |unet_image| image:: https://github.com/qubvel/segmentation_models/blob/master/images/unet.png
.. |linknet_image| image:: https://github.com/qubvel/segmentation_models/blob/master/images/linknet.png
.. |psp_image| image:: https://github.com/qubvel/segmentation_models/blob/master/images/pspnet.png
.. |fpn_image| image:: https://github.com/qubvel/segmentation_models/blob/master/images/fpn.png

**Backbones**

.. table:: 

    =============  ===== 
    Type           Names
    =============  =====
    VGG            ``'vgg16' 'vgg19'``
    ResNet         ``'resnet18' 'resnet34' 'resnet50' 'resnet101' 'resnet152'``
    SE-ResNet      ``'seresnet18' 'seresnet34' 'seresnet50' 'seresnet101' 'seresnet152'``
    ResNeXt        ``'resnext50' 'resnext101'``
    SE-ResNeXt     ``'seresnext50' 'seresnext101'``
    SENet154       ``'senet154'``
    DenseNet       ``'densenet121' 'densenet169' 'densenet201'`` 
    Inception      ``'inceptionv3' 'inceptionresnetv2'``
    MobileNet      ``'mobilenet' 'mobilenetv2'``
    EfficientNet   ``'efficientnetb0' 'efficientnetb1' 'efficientnetb2' 'efficientnetb3' 'efficientnetb4' 'efficientnetb5' efficientnetb6' efficientnetb7'``
    =============  =====

.. epigraph::
    All backbones have weights trained on 2012 ILSVRC ImageNet dataset (``encoder_weights='imagenet'``). 


Installation
~~~~~~~~~~~~

**Requirements**

1) python 3
2) keras >= 2.2.0 or tensorflow >= 1.13
3) keras-applications >= 1.0.7, <=1.0.8
4) image-classifiers == 1.0.*
5) efficientnet == 1.0.*

**PyPI stable package**

.. code:: bash

    $ pip install -U segmentation-models

**PyPI latest package**

.. code:: bash

    $ pip install -U --pre segmentation-models

**Source latest version**

.. code:: bash

    $ pip install git+https://github.com/qubvel/segmentation_models
    
Documentation
~~~~~~~~~~~~~
Latest **documentation** is avaliable on `Read the
Docs <https://segmentation-models.readthedocs.io/en/latest/>`__
