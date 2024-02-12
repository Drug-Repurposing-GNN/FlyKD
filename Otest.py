import noisy_student

pLabelsWscore = noisy_student.generate_psuedo_labels()

pLabelsWscore.to_csv('psuedo_labels_w_scores.csv')
