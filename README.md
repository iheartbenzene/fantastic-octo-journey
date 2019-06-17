# The Intelligent Budget

A budgeting program to track and analyze expenses.

### Getting Started

This project uses `python 3.7.x` and `jupyter notebooks`.

##### To install jupyter and python 3.7.x:

+ Anaconda: https://www.anaconda.com/distribution/#download-section
+ Pandas: https://github.com/pandas-dev/pandas or https://pandas.pydata.org/
+ Jupyter Notebook: https://jupyter.org/install

Then select the `Pyghon 3.7 version` download link.

```
Anaconda installs python 3.7.x as well as the jupyter notebook in one fell swoop.
Note: It should automatically detect the operating system for Windows and Mac 
      but might have issues with Linux; in which case manually select that tab 
      before proceeding.
```

##### Clone the repo:

SSH: `git clone git@github.com:iheartbenzene/fantastic-octo-journey.git && cd fantastic-octo-journey`

HTTPS: `git clone https://github.com/iheartbenzene/fantastic-octo-journey.git && cd fantastic-octo-journey`

Load the notebook:
+ `budget_mk_X.ipynb` : https://github.com/iheartbenzene/fantastic-octo-journey/blob/master/budget_mk_X.ipynb

or to both clone and load the notebook:
```
  git clone git@github.com:iheartbenzene/fantastic-octo-journey.git

  cd fantastic-octo-journey

  jupyter notebook
```
+ Click `budget_mk_X.ipynb` once the notebook has loaded.
___
Thanks to Pandas and the Google developer console, these are some of the results of the analysis.

![Most Visited Merchants](https://i.imgur.com/G5OK1Zc.png)
![Occurrences by Merchant Type](https://i.imgur.com/WxzTfIh.png)
![Ocurrences by Purchase Type](https://i.imgur.com/52AznJf.png)
![Ocurrences by Merchant](https://i.imgur.com/nhSnPJ3.png)
![Annual Budget](https://i.imgur.com/FTWVyjp.png)
___

```
To use the notebook on yourself, insert your key value in the authentication line.
After which, it would become possible to import your spreadsheet and have the system update 
automatically to perform your own analysis.
```

***

There exists a prototype written in GNU Octave. However, that that was just a proof of concept and was left here mainly as a trail for anyone else who would like to follow along.

To use the Octave/Matlab edition on yourself, add or update the files called "work_hours.txt", "work_week.txt", "bills_01.txt", "hourly_rate.txt".

___

To acquire GNU Octave: 

1. In your web browser, go to https://www.gnu.org/software/octave/

2. Click "download".

3. Select your operating system.

Windows: 

If you have a newer computer, choose Windows-64 (recommended) || octave-5.1.0-w64-installer.exe (~ 286 MB)

Linux:

1. Open a terminal instance.

2. Type the command apt install octave

&nbsp;&nbsp;&nbsp;&nbsp; 2b. Use the command, sudo apt install octave, if you get a permissions error.

***
