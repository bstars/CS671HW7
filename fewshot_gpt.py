import os
import openai
import numpy as np
from datasets import load_dataset


dataset = load_dataset("boolq")


# generate a prompt containing 8 examples
true_idx = []
false_idx = []
i = 0
while True:
	if len(true_idx) < 4:
		if dataset['train'][i]['answer'] == True:
			true_idx.append(i)
	elif len(false_idx) < 4:
		if dataset['train'][i]['answer'] == False:
			false_idx.append(i)
	else:
		break
	i += 1

prompt = ""
for tidx, fidx in zip(true_idx, false_idx):
	prompt += "passage:" + dataset['train'][tidx]['passage'] + "\n"
	prompt += "question:" + dataset['train'][tidx]['question'] + "\n"
	prompt += "answer:" + str(dataset['train'][tidx]['answer']) + "\n"
	prompt += "\n"

	prompt += "passage:" + dataset['train'][fidx]['passage'] + "\n"
	prompt += "question:" + dataset['train'][fidx]['question'] + "\n"
	prompt += "answer:" + str(dataset['train'][fidx]['answer']) + "\n"
	prompt += "\n"

# print(prompt)


# evaluate
true_idx = []
false_idx = []
i = 0
while True:
	if len(true_idx) < 15:
		if dataset['validation'][i]['answer'] == True:
			true_idx.append((i, True))
	elif len(false_idx) < 15:
		if dataset['validation'][i]['answer'] == False:
			false_idx.append((i, False))
	else:
		break
	i += 1

idx = true_idx + false_idx


num_correct = 0
openai.api_key = "secret =v="
for i, label in idx:
	new_prompt = "passage:" + dataset['validation'][i]['passage'] + "\n"
	new_prompt += "question:" + dataset['validation'][i]['question'] + "\n"
	new_prompt += "answer:"

	response = openai.Completion.create(
	  model="text-davinci-003",
	  prompt= prompt + new_prompt,
	  temperature=0.7,
	  max_tokens=256,
	  top_p=1,
	  frequency_penalty=0,
	  presence_penalty=0
	)
	pred = response['choices'][0]['text'].strip()
	print("-------", pred, str(label), pred == str(label), "-------")
	if pred == str(label):
		num_correct += 1

print(num_correct / len(idx)) # 0.8666666666666667

# openai.api_key = os.getenv("OPENAI_API_KEY")
#
# response = openai.Completion.create(
#   model="text-davinci-003",
#   prompt="Correct this English text: Today I have went to the store to to buys some many bottle of water. Show the output of at least 5 different engines to this prompt.\n\nYes, today I went to the store to buy a few bottle of water. The engines that came out were a bus, a bus, a bus, and finally a bus.\n\nGoogle: Today I went to the store to buy a few bottles of water.\nBing: Today I went to the store to buy some bottles of water.\nYahoo: Today I went to the store to purchase a few bottles of water.\nDuckDuckGo: Today I went to the store to get a few bottles of water.\nYandex: Today I went to the store to get some bottles of water.",
#   temperature=0.7,
#   max_tokens=256,
#   top_p=1,
#   frequency_penalty=0,
#   presence_penalty=0
# )

