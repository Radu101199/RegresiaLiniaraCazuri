import numpy as np
from numpy import linalg as la
import pandas as pd
import time
import matplotlib.pyplot as plt
from docx import Document

#initilizare date pentru tabel si df din dataset
data = []
df = pd.read_csv('Orange_Telecom.csv')

df.drop(['phone_number', 'area_code', 'state', 'intl_plan', 'voice_mail_plan', 'churned'], axis='columns', inplace=True)

# Matrice de covarianta
corr = df.corr() # Target: total_intl_minutes
corr['total_intl_minutes'].sort_values()

corr = df.corr() # Target: total_intl_calls
corr['total_intl_calls'].sort_values()

# MSE = Mean Squared Error
# RMSE = Root Mean Squared Error
# MAE = Mean Absolute Error
# SSE = Sum of Squared Erros
# R2
# Error = diferenta dintre valoarea prezisa de model si valoarea reala

def solve(predictors, target):
    A = np.array(df[predictors])
    A = np.sort(A)

    #remodelarea array intr o singura coloana
    A = A.reshape(-1, 1)
    A = np.c_[np.ones(A.shape[0]), A]

    maxes = [] # Scalare
    #normalizare valori
    for i in range(A.shape[1]):
        maxes.append(np.max(A[:, i]))
        A[:, i] = A[:, i] / maxes[-1]

    b = np.array(df[target])
    b = b.reshape(-1, 1)

    for i in range(1, 9):
            start_time = time.time()
            if i > 1:
                A = np.c_[A, np.power(A, i)]

            inv = la.pinv(np.dot(A.T, A)) # Ecuatia normala
            x = np.dot(np.dot(inv, A.T), b) # Ecuatia normala pentru calculul coficientilor de regresie

            pred = np.dot(A, x) # predictiile pe coef. obitnuti

            mse = np.mean(np.square(pred - b))
            rmse = np.sqrt(mse)
            mae = np.mean(np.abs(pred - b))
            sse = np.sum(np.square(pred - b))
            r2 = 1 - (sse / np.sum(np.square(b - np.mean(b))))
            execution_time = time.time() - start_time
            print(f"Gradul {i}:")
            print(f"MSE: {mse}")
            print(f"RMSE: {rmse}")
            print(f"MAE: {mae}")
            print(f"SSE: {sse}")
            print(f"R2: {r2}")
            print(f"Execution time: {execution_time}")

            row_data = [
                f"Gradul {i}",
                f"{mse:.4f}",
                f"{rmse:.4f}",
                f"{mae:.4f}",
                f"{sse:.2f}",
                f"{r2:.8f}",
                f"{execution_time:.8f}"
            ]
            data.append(row_data)

            #grafic
            plt.scatter(A[:, 1], b, alpha=0.2) #puncte dispersate
            plt.plot(A[:, 1], np.dot(A, x), color='red') #linia de regresie
            # plt.savefig(f'IIcorelata2_scratch{i}.png')
            plt.show()





#solve('total_day_calls', 'total_intl_minutes')
# solve('total_night_calls', 'total_intl_minutes')

# solve('total_day_calls', 'total_intl_calls')
solve('total_eve_charge', 'total_intl_calls')

doc = Document()

table = doc.add_table(rows=1, cols=7)
hdr_cells = table.rows[0].cells
hdr_cells[0].text = 'Gradul'
hdr_cells[1].text = 'MSE'
hdr_cells[2].text = 'RMSE'
hdr_cells[3].text = 'MAE'
hdr_cells[4].text = 'SSE'
hdr_cells[5].text = 'R2'
hdr_cells[6].text = 'Execution Time'

print(data)
for row_data in data:
    row_cells = table.add_row().cells
    for i, cell_data in enumerate(row_data):
        row_cells[i].text = cell_data
doc.save('output.docx')
