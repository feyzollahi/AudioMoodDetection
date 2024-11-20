Hello

This project is about audio processing.

Dataset comes from Sharif speech dataset.

You can run the demo with these 3 steps:

First install the requirements:

```bash
pip install -r requirements.txt
```
for training, we should extract some features from audio. for this purpose, we used [egemaps version 2](https://audeering.github.io/opensmile-python/api/opensmile.FeatureSet.html#egemapsv02).

then you should perform the training of the neural network with audio data
```bash
python src/codes/trainData.py
```

This may take a few minutes! (up to 30 minutes)

finally you can test the model:
```bash
python src/codes/demo.py
```

This demo shows the probability for the mood and gender:

### moods

"**Angry**", "**Fearful**", "**Happy**", "**Sad**", "**Neutral**", "**Surprise**"

### gender

"**Male**", "**Female**"

Please add your voices to the **src/resource/voices** folder