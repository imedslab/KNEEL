# KNEEL: Hourglass Networks for Knee Anatomical Landmark Localization

<center>
<img src="pics/landmarks_kl.png" width="800"/>
</center>


(c) Aleksei Tiulpin, University of Oulu, 2019-2024

## What this repo is about
This repo contains an inference package for the models trained in or paper KNEEL: https://github.com/Oulu-IMEDS/KNEEL. In that paper, we have developed a neural network architecture, which allows to accurately detect knee anatomical landmarks, and have validated the model on several datasets.

<center>
<img src="pics/network_arch.png" width="800"/> 
</center>

In this repo, we have included a web-app, which is dockerized, and can be accessed via http protocol.


## Before starting
The very first step to access our model, is getting access to the Hugging Face repo: https://huggingface.co/imeds/kneel. You can request an automatically approved access. Subsequently, in your settings generate a token. More on this can be found in the documentation of HuggingFace: https://huggingface.co/docs/hub/en/security-tokens.

The token is required to get the KNEEL app running.

## Running the KNEEL app

You need to have docker installed. If you want to use GPU, you must have the GPU runtime installed as well. Below is how you can run the code:

On CPU (slow, but works on all )
```
docker run -it --name kneel_api_cpu --rm \
  -v $(pwd)/tmp:/tmp/:rw -p 5000:5000 --ipc=host \
  imeds/kneel:cpu python -u -m kneel.api.app \
  --refine --jit_trace --deploy --device cpu \
  --hf_token <YOUR_HUGGING_FACE_TOKEN>
```

On GPU (a lot faster)
```
docker run -it --name kneel_api_gpu --rm --runtime=nvidia --gpus all\
  -v $(pwd)/tmp:/tmp/:rw -p 5000:5000 --ipc=host \
  imeds/kneel:gpu python -u -m kneel.api.app \
  --refine --jit_trace --deploy --device cuda:0 \
  --hf_token <YOUR_HUGGING_FACE_TOKEN>
```

**Note:** If you want to see the full logs, do
```
tail -f tmp/kneel.log
```

## Making predictions

To make predictions, just send a POST request with a json having `{"dicom":<RAW_DICOM_IN_BASE_64>}` to `/kneel/predict/bilateral`. To encode a DICOM image in Python, read it as a binary file and then use standard python base64 library: `base64.b64encode(dicom_binary).decode('ascii')` to generate a base64 string. You can do this as follows (assuming that the microservice runs on `localhost`):

```
import requests
...
...
with open(img_path, "rb") as f:
    data_base64 = base64.b64encode(f.read()).decode('ascii')
response = requests.post("http://localhost/kneel/predict/bilateral", json={'dicom': data_base64})
```
As a result, you will get an array of 16 anatomical landmarks in (x, y) format. Their meaning can be seen in the paper, Figure 1.

## Customizing 
If any new dependencies are added, you can recompile the dockers as follows (from the main repo directory)
```
docker buildx build -t imeds/kneel:cpu -f docker/Dockerfile.cpu .
docker buildx build -t imeds/kneel:gpu -f docker/Dockerfile.gpu .
```

## License & citations
You must cite the following paper (Accepted to ICCV 2019 VRMI Workshop).

```
@inproceedings{9022083,
  author={Tiulpin, Aleksei and Melekhov, Iaroslav and Saarakkala, Simo},
  booktitle={2019 IEEE/CVF International Conference on Computer Vision Workshop (ICCVW)}, 
  title={KNEEL: Knee Anatomical Landmark Localization Using Hourglass Networks}, 
  year={2019},
  volume={},
  number={},
  pages={352-361},
  doi={10.1109/ICCVW.2019.00046}
}
```

The codes and the pre-trained models are not available for any commercial use 
including research for commercial purposes.
