# Once-For-All Scale Super-resolution (OFA-SR)
This is a brief introduction about how to use this code.

Please refer to [Meta-SR](https://github.com/XuecaiHu/Meta-SR-Pytorch) to prepare the training and testing dataset.

* overall-framework
![](/fig/overall-framework.png)

# Useful model
- Full-net: [multi_rdn_metashuffle_v7.py](https://github.com/liangheng96/OFA-SR/blob/master/model/multi_rdn_metashuffle_v7.py) gets the better performance than Meta-SR in all scale factor.

- Sub-net: [multi_srdn_metashuffle_v10.py](https://github.com/liangheng96/OFA-SR/blob/master/model/multi_srdn_metashuffle_v10.py) gets the best simmable sub-net.


# Training
- Please refer to [train_demo](https://github.com/liangheng96/OFA-SR/blob/master/config_demo/train_demo.sh) to get the training configuration.


# Testing
- Please refer to [Test_demo](https://github.com/liangheng96/OFA-SR/blob/master/config_demo/test_demo.sh) to get the training configuration.


# Current Performance
* Full-net
![](/fig/full-net.png)

* Sub-net
![](/fig/sub-net.png)



# Help
If you find any problem to use this code, please ask me for help.

My email: henrycliang@gmail.com

Wechat: heng52028