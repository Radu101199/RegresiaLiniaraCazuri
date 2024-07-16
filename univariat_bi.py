import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from docx import Document

df = pd.read_csv('Orange_Telecom.csv')
data =[]
def solve(predictors, target):
    A = np.array(df[predictors])
    A = np.sort(A)

    # remodelarea array intr o singura coloana
    A = A.reshape(-1, 1)
    A = np.c_[np.ones(A.shape[0]), A]

    maxes = []

    #normalizarea coloanelor
    for i in range(A.shape[1]):
        maxes.append(np.max(A[:, i]))
        A[:, i] = A[:, i] / maxes[-1]

    b = np.array(df[target]).reshape(-1, 1)

    for i in range(1, 9):
        start_time = time.time()
        if i > 1:
            A = np.c_[A, np.power(A, i)]

        lr = LinearRegression()
        lr.fit(A, b) #antrenare
        pred = lr.predict(A) #predictie
        
        mse = mean_squared_error(b, pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(b, pred)
        sse = np.sum(np.square(pred - b))
        r2 = r2_score(b, pred)
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

        plt.scatter(A[:, 1], b, alpha=0.2)
        plt.plot(A[:, 1], np.dot(A, np.insert(lr.coef_[0, 1:], 0, lr.intercept_)), color='red')
        # plt.savefig(f'corelata1univ_bi{i}.png')
        plt.show()
        
# solve('total_day_calls', 'total_intl_minutes')
solve('total_night_calls', 'total_intl_minutes')

#solve('total_day_calls', 'total_intl_calls')
# solve('total_eve_charge', 'total_intl_calls')

# Create a new Word document
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

# Add data to the table
print(data)
for row_data in data:
    row_cells = table.add_row().cells
    for i, cell_data in enumerate(row_data):
        row_cells[i].text = cell_data
# Save the document
doc.save('output.docx')