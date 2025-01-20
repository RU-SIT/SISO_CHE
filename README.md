# MyPrivaterepo

Welcome to this repository where we are exploring some new aspects of machine .ealrning in wireless communication domain. The main focuse of this work is SISO-OFDM Channel Estimation.

To run an experiment you need to download the data [here](https://drive.google.com/drive/folders/10hIn854d219OhhC7hLnUOUD_6revp3U1?usp=sharing). Then follow the below steps:
1- Run channel2D.py to create a dictionary of data
2- To train the meta model, run MAML_trainer.py (you can set arguments eigther in the script or in the command line)
3_ By running MAML_finetuning.py, you can fine tune the model and save the finetuned model and evalution.
4_ If you need to compare the results, then run test.py.

