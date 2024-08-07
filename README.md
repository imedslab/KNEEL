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




## Running the KNEEL app

You need to have docker installed. If you want to use GPU, you must have the GPU runtime installed as well. Below is how you can run the code:

On CPU (slow, but works on all )
```
docker run -it --name kneel_api_cpu -v $(pwd)/tmp:/tmp/:rw -v -p 5000:5000 --ipc=host imeds/kneel:cpu python -u -m kneel.api.app --refine --jit_trace --deploy --device cpu --hf_token --hf_token <YOUR_HUGGING_FACE_TOKEN>
```

On GPU (a lot faster)
```
docker run -it --name kneel_api --rm --runtime=nvidia --gpus all -v $(pwd)/tmp:/tmp/:rw -p 5000:5000 --ipc=host imeds/kneel:gpu python -u -m kneel.api.app --refine --jit_trace --deploy --device cuda:0 --hf_token <YOUR_HUGGING_FACE_TOKEN>
```

Just send a POST request with a json having `{"dicom":<RAW_DICOM_IN_BASE_64>}` to `/kneel/predict/bilateral`. To encode a DICOM image in Python,
just read it as a binary file and then use standard python base64 library: `base64.b64encode(dicom_binary).decode('ascii')`. You can do this as follows (assuming that the microservice runs on `localhost`):

```
with open(img_path, "rb") as f:
    data_base64 = base64.b64encode(f.read()).decode('ascii')
response = requests.post(args.kneel_addr + "/kneel/predict/bilateral", json={'dicom': data_base64})
```
## Customizing 
If any new dependencies are added, you can recompile the dockers as follows (from the main repo directory)
```
docker buildx build -t imeds/kneel:cpu -f docker/Dockerfile.cpu .
docker buildx build -t imeds/kneel:gpu -f docker/Dockerfile.gpu .
```

## License
If you use the annotations from this work, you must cite the following paper (Accepted to ICCV 2019 VRMI Workshop)

```
@article{tiulpin2019kneel,
  title={KNEEL: Knee Anatomical Landmark Localization Using Hourglass Networks},
  author={Tiulpin, Aleksei and Melekhov, Iaroslav and Saarakkala, Simo},
  journal={arXiv preprint arXiv:1907.12237},
  year={2019}
}
```

The codes and the pre-trained models are not available for any commercial use 
including research for commercial purposes.
