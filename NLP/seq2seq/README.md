This model is a simple model based on RNN plus cross  attention mechanism. This model is a demo for a simple question answering system.

Word bag/ Dataset: answer.txt, question.txt

The model can be seen in rnn.py

Loss for our RNN model

<img src="./pic/loss.png"></img>

Attention Map for our result:

Question: What time does the next tram to the park depart

Model output: is scheduled departs at 10:30 AM...

The corresponding attention map

<img src="./pic/att.png"></img>