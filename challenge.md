# Notes

SPAGHETTI-1 has had a rough couple of weeks. Do not assume that the training archive and the new observations look alike, or that every entry in the archive can be trusted.

The classes are not balanced: some pasta shapes are much rarer than others.

# Desc
Milan Space Center, 14 days before the launch.

The Italian Space Agency has finally deployed SPAGHETTI-1 (Space Photography And Galactic High-resolution Exploration Telescope Technology Initiative), a revolutionary telescope designed to classify stars across the galaxy. Everything was going perfectly. Thousands of stellar images were streaming in, the scientists were celebrating, and the espresso machine was working overtime.

Until something strange started happening. Unexplained visual artifacts, bizarre color distortions, and suspiciously systematic disturbances began appearing in the telescope's images. The engineers blame faulty equipment. The astronomers blame cosmic radiation. Meanwhile, someone in Mission Control has noticed an uncanny resemblance to certain classified Area 51 files.

Coincidence? Probably.

Or at least, that's what they want you to believe.

Who's they, you ask? The shadow government, obviously. Keep up.

With just 14 days left, your mission is to classify the stars, identify corrupted observations, and uncover whatever is interfering with the telescope. Trust the data. Question the noise. And keep watching the stars.

# Leaderboard Metrics

In this assignment, you will receive a collection of 256 × 256 RGB images captured by SPAGHETTI-1. The objective of this task is image classification — that is, to assign each image to one of ten possible classes.

Your models will be evaluated according to the weighted F1-Score. For each class,
$$ \text{F1}_c=\frac{2 \times \text{Precision}_c \times \text{Recall}_c}{\text{Precision}_c + \text{Recall}_c=\frac{\text{TP}_c}{\text{TP}_c + \frac{1}{2}(\text{FP}_c + \text{FN}_c)}$$
and the final score averages the ten classes, each weighted by its share of the test images:
$$ \text{F1}_{weighted}=\sum_{c=0}^{9}\frac{n_c}{N}\text{F1}_c$$
where $n_c$ is the number of test images of class c and N is the number of all test images.

To prevent overfitting, during the entire competition period your submissions will be evaluated on a specific subset of the test set. The final results evaluated on a separate test subset will be available at the end of the challenge.

# Submission file
A `.csv` file in format:
```csv
id,label
test_00000,3
test_00001,7
test_00002,0
etc.
```

# Report
Before the end of the challenge, you are required to submit a report detailing your activities. The report should include:
- Information about team members.
- The development process of your model.
- The ideas behind your final solution.
- All necessary information to demonstrate your work.

The final report is part of your challenge grade, alongside your results from the Leaderboard.

The report must be a PDF file with a maximum length of 3 pages excluding references. It is mandatory to use the template provided here: [LaTeX Template on Overleaf](https://www.overleaf.com/project/6b04bff3f198011de6fb3958/share#13c2310504d6bfed83ddb0a7b03666d1d0ef3d2e20ea59b6). This is to ensure a cohesive structure for all your reports. Please note that any change to the template and any additional page after the 3 allowed will result in penalties.

To submit your report, one member of the team must send an email to airlab.official.polimi@gmail.com. Attach the report to this email.

In addition to the report, you must attach to the email a zip file containing one (or more) notebook(s) that precisely demonstrate the models described in the report. This includes the code for creating and training the models, along with the local performance results, visible in the notebook output cells. The notebooks must be fully executed, with all outputs visible. Ensure that your code is well-commented and reproducible, highlighting the roles of functions and variables, as well as the main steps.

# Turkey Game

[Google Docs](https://docs.google.com/document/d/1cT5GYReh-sJcn__1TwDsN3SzjSfok1MXo9M94wS7K70/edit?usp=sharing)
