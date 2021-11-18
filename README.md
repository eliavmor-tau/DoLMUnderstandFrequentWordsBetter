# Do Language Models Understand Frequnt Words Better?

This work was initially inspired by Talmor's work
## Abstract 
Pre-trained  language  models  (LMs)  have  re-cently demonstrated outstanding results acrossa variety of tasks. However, it remains unclearprecisely what knowledge the LM manages tocapture during pre-training and how word fre-quency in the training corpus affects the acqui-sition of knowledge about these words. In thiswork we explore the correlation between wordoccurrence in the language and general worldknowledge acquired by a Language Model dur-ing  pre-training.We  propose  a  frameworkfor  testing  this  subject,  using  a  downstream”Yes/No”  QA  task.   The  model  (in  our  caseT5) is trained on a large set of questions andthen  tested  on  its  knowledge  of  relations  be-tween  different  animals  and  their  properties,e.g.    ”Does  a  fish  have  wings?”.    Our  find-ings  show  positive  correlations  between:  (a)word occurrence and the accuracy of answersfor this word. (b) co-occurrence of two wordsand a tendency of the model to answer ”Yes”for questions about their relation. We also cre-ated  a  classifier  for  predicting  the  answer  ofthe  model  for  animal/property  relation  ques-tions based on the co-occurrence rate of thesetwo words.


## References
@misc{talmor2020olmpics,
      title={oLMpics -- On what Language Model Pre-training Captures}, 
      author={Alon Talmor and Yanai Elazar and Yoav Goldberg and Jonathan Berant},
      year={2020},
      eprint={1912.13283},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
<a id="1">[1]</a> 
Dijkstra, E. W. (1968). 
Go to statement considered harmful. 
Communications of the ACM, 11(3), 147-148.
