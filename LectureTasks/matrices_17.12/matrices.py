Row = int(input("Enter the number of rows: "))
Column = int(input("Enter the number of columns: "))
        # Funktion zur Erstellung einer neuen Matrix
def NewMatrix(x, y):
    
    print(f"Enter the entries for Matrix {x} row-wise:")
    matrix = []
    for row in range(y):  # 'y' steht für die Anzahl der Zeilen
        a = []
        for column in range(y):  # 'y' steht auch für die Anzahl der Spalten
            a.append(int(input()))
        matrix.append(a)

    # Matrix ausgeben
    print(f"Matrix {x}:")
    for row in matrix:
        for column in row:
            print(column, end=" ")
        print()
    return matrix

# Zwei Matrizen erstellen und ausgeben
matrix1 = NewMatrix(1, Row)  # Erste Matrix
matrix2 = NewMatrix(2, Row)  # Zweite Matrix
