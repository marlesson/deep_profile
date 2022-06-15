# Deep Profile

## install

pip install -r requirements.txt

### Dependences

pip install imutils
pip install git+https://github.com/ageitgey/face_recognition_models
pip install dlib
python -m pip install opencv-python
pip install omegaconf
pip install keras==2.0.3
pip install tensorflow-gpu==1.15.2

## Uso

Download dos pesos do modelo. (Modelo já treinado)

```sh
mkdir -p pretrained_models
wget -P pretrained_models https://github.com/yu4u/age-gender-estimation/releases/download/v0.5/weights.28-3.73.hdf5
```

Executa a webcan

```sh
python3 webcam.py
PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python3 webcam.py --sleep 0.1 --ignore-gender
```


Executar para Imagens

```sh
python predict_image.py
```


## Referências

* Código baseado em https://github.com/yu4u/age-gender-estimation
