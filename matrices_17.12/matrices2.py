import pprint
import random
#Functions
def CreateMatrix():
    Row = int(input("Enter the number of rows:"))
    Column = int(input("Enter the number of columns:"))

    # Initialize matrix
    matrix = []
    print("Enter the entries row wise:")

    # For user input
    # A for loop for row entries
    for row in range(Row):
        a = []
        # A for loop for column entries
        for column in range(Column):
            a.append(int(input()))
        matrix.append(a)
    return matrix
# Initialize matrix with random numbers
def CreateMatrixRandom(Rows, Columns):
    
    matrix = [[random.randint(0, 10) for _ in range(Columns)] for _ in range(Rows)]
    return matrix

def PrintMatrix(matrix):
    pprint.pprint(matrix)

#Validate List Dimensions
def ListSizeEquals(listA, listB):
    rowCountA = len(listA)
    rowCountB = len(listB)
    return rowCountA == rowCountB

#validate matrix dimensions
def ValidateDimensions(matrixA, matrixB):
    rowsMatch = ListSizeEquals(matrixA, matrixB)
    columnA = matrixA[0]
    columnB = matrixB[0]
    return rowsMatch and ListSizeEquals(columnA, columnB)

#For adding the matrix
def AddMatrices(matrixA, matrixB):
    newMatrix = []
    if ValidateDimensions(matrixA, matrixB):
        for rowIndex in range(len(matrixA)):
            newRow = []
            for columnIndex in range(len(matrixA[0])):
                newRow.append(matrixA[rowIndex][columnIndex] + matrixB[rowIndex][columnIndex])
            newMatrix.append(newRow)
    return newMatrix

#For adding the matrix
def SubtractMatrices(matrixA, matrixB):
    newMatrix = []
    if ValidateDimensions(matrixA, matrixB):
        for rowIndex in range(len(matrixA)):
            newRow = []
            for columnIndex in range(len(matrixA[0])):
                newRow.append(matrixA[rowIndex][columnIndex] - matrixB[rowIndex][columnIndex])
            newMatrix.append(newRow)
    return newMatrix                   

def MultiplyMatricesWithSwapping(matrixA, matrixB):
    newMatrix = []
    
    if ListSizeEquals(len(matrixA) != len(matrixB[0])):
        if ListSizeEquals(len(matrixB) != len(matrixA[0])):
            return None
        else:  # Multiplication is possible the other way around -> Need to swap
            buffer = matrixA
            matrixA = matrixB
            matrixB = buffer
    
    # Initialize newMatrix with zero values
    for i in range(len(matrixA)):
        newRow = []
        for j in range(len(matrixB[0])):
            newRow.append(0)
        newMatrix.append(newRow)
    
    # Perform the multiplication
    for i in range(len(matrixA)):
        for j in range(len(matrixB[0])):
            for k in range(len(matrixB)):
                newMatrix[i][j] += matrixA[i][k] * matrixB[k][j]
                
    return newMatrix

def MultiplyMatrices(matrixA, matrixB):
    # Verify dimensions: Columns of A must match Rows of B
    if len(matrixA[0]) != len(matrixB):
        return None  # Invalid multiplication
    
    # Initialize result matrix with zeros
    result = [[0 for _ in range(len(matrixB[0]))] for _ in range(len(matrixA))]
    
    # Perform multiplication: 
    for firstMatrixRow in range(len(matrixA)):           # 
        for SecondmatrixColumns in range(len(matrixB[0])):    # Iterate over columns of B
            for SecondMatrixRows in range(len(matrixB)):   # Iterate over rows of B (or columns of A)
                cellValue = matrixA[firstMatrixRow][SecondMatrixRows] * matrixB[SecondMatrixRows][SecondmatrixColumns]
                result[firstMatrixRow][SecondmatrixColumns] += cellValue
    return result

        
    
        
#Script:
#MatA = CreateMatrix()
#MatB = CreateMatrix()
MatC = CreateMatrixRandom(3,5)
PrintMatrix(MatC)
MatD = CreateMatrixRandom(5,3)
PrintMatrix(MatD)
#MatResult = AddMatrices(MatA, MatB)
MatResult = MultiplyMatrices(MatC, MatD)
PrintMatrix(MatResult)


input("Press Enter to exit...")
