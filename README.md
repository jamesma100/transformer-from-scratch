# transformer-from-scratch

Pytorch implementation of the original transformer model described in the 2017 [Attention is all you need](https://arxiv.org/abs/1706.03762) paper. This is mostly for educational purposes so I made basically no attempt to optimize for efficiency. I also wrote a simple tokenizer based on [byte pair encoding](https://en.wikipedia.org/wiki/Byte_pair_encoding).

## Overview
- `tokenizer.py` - tokenizer for input text
- `transformer.py` - actual model is defined here
- `train.py` - script for training model
## Setup

```
$ python3 -m venv venv
$ source ./venv/bin/activate
$ pip install -r ./requirements.txt
```

## Training
You should be able to run this on any corpus but as an example, we'll train the model on Samurai Champloo English subtitles.

Run tokenizer until a vocabulary size of 8000 is reached:
```
$ python3 tokenizer.py ./data/samurai_champloo_cleaned 8000
```
This will write output to files under `./out` where:
- `tokens.txt`: token list generated from the corpus
- `base.json`: base vocabulary, i.e. distinct tokens already present in corpus
- `vocab.json`: actual generated vocabulary (should have size 8000)

Then run the train script to train the model:
```
$ ./python3 train.py
```
You should see the per-epoch loss printed to the terminal:
```
[INFO] epoch loss:  tensor(9.0069, grad_fn=<DivBackward0>)
[INFO] epoch loss:  tensor(9.0106, grad_fn=<DivBackward0>)
[INFO] epoch loss:  tensor(9.0000, grad_fn=<DivBackward0>)
...
```

At the end it should generate a small sample of text. It should be mostly gibberish
```
understand ata yo ga akereba owakare Yume wa tooki maboroshi ni Anata o oikakete ita hikari no naka de dakareru tabi Atatakai kaze o tayori As I was bathed in the light that followed on your heels Haru o tsuge odoridasu sansai Natsu o miru Uji nohara karakusa kawaku wa Aki no tsuki nobotta manmarusa o-iwai The autumn moon rises, let's celebrate its fullness Fuyu o sugi mata tsukihi o kazoeru Mada mabuta no oku ni aru itsuka no natsu toosugita aozora (Atatakakatta) Te o tsunagu hana tsukamiutaa trap. give you place, where. worry so warm than en  to my collect red to mewandhelp you?! Get interbelieve . I see. is the way  these guys om guys, ... Thus to . Can you filthyakuza , they say  to twoloud that you're sworder and In other words, we travel westwards, wandering hoodlums who think we 're samurai The souls accompanying us drift up to the sky If they flget movingbecome wice call me top these , damn won't be , don't you thinkmade  these guys awn comes, and we part ways once again My dreams becoming distant apparitions I turn to the warm wind for help, the wind I felt every time you held me alkfor vase... Te to ll rightget help piapprecibottom travwhereoffifter thatooking down from up there, we wouldn't look any bigger than gr thenlucka minut? Okuruing it settled willguyo gsunset out to dry The autumn moon rises, let's celebrate its fullness Winter passes by, and I count off the days and months again my eyes  three . It's justsaying that awn comes, and we part ways once again My dreams becoming distant apparhave youbabeneralleaves ing in ata tsukihi happened  turn to the warm wind for help, the doesn't mean . You. We oter in ed to the ?! Wup to the hootpe. We ely . I think s of your? What's riverata yo ga akereba owakare Yume wa tooki maboroshi ni Anata o oikakete ita hikari no naka de dakareru tabi Atatakai kaze o tayori As I was bathed in the light that followed on your heels Haru o tsuge odoridasu sansai Natsu o miru Uji nohara karakusa kawaku wa Aki no tsuki nobotta manmarusa o-iwai The autumn moon rises, let's celebrate its fullness Fuyu o sugi mata tsukihi o kazoeru Mada mabuta no oku ni aru itsuka no natsu toosugita aozora (Atatakakattabilllaw peopleouble something in particulstrange cloth?! That In other words, we travel westwards, wandering hoodlums who think we 're samurai The souls accompanying us drift up to the sky If they flap their wings, they complpolice officers with the Matsumae domain magistrate's it out meripay vase...
```

## Final thoughts
At the moment the model doesn't really work since the cross entropy doesn't decrease after a certain point, even when I try to overfit using a small training set. There might be errors in the code, or maybe it just needs to train for longer? I'm also running this on a laptop CPU (lol). It was also hard to figure out what values should be used for starting parameters. It's possible that the model will converge faster if the following were adjusted:
- `ctx_sz`: context size
- `batch_sz`: batch size
- `lr`: learning rate
- `vocab_sz`: vocab size
- `d_model` dimension of the input model; the input is (`batch_sz`, `ctx_sz`, `d_model`)

The paper includes all these parameters, but it didn't really touch on how they were derived or what they should be given smaller training data.

Apparently you're also supposed to tie the weights of the unembedding layer to the transpose of the weights of the initial embedding layer, a practice known as [weight tying](https://paperswithcode.com/method/weight-tying), which I didn't bother to implement.

All in all, I felt like transformers in their current state aren't really grounded in math theory and a lot of it is just empirically tweaking things until it eventually works without really understanding why. But alas, if there are glaring errors you find in here or proposals, feel free to email me, or better yet, open a PR!
