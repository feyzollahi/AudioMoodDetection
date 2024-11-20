Hello

This project is about audio processing.

Dataset if from sharif speech dataset.

You execute demo with these 3 steps:
first install the requirements:

```bash
pip install -r requirements.txt
```

then you should execute training of neural network with audio data
```bash
python src/codes/trainData.py
```

It could last several minutes!

finally you can test the model:
```bash
python src/codes/demo.py
```

This demo shows the probability of the mood and gender:

### moods

"**Angry**", "**Fear**", "**Happy**", "**Sad**", "**Neutral**", "**Surprise**"

### genders

"**Male**", "**Female**"

Please paste your voice to **src/resource/voices** folder