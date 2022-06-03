
# auxiliary functions:


def range_creator(requested_p):

    closest_number = min(xVectors, key=lambda z: abs(z - requested_p))
    if closest_number < requested_p:
        return closest_number
    else:
        return closest_number - 1


def matrix_multiply(a, b):  # A function that calculates the multiplication of 2 matrices and returns the new matrix
    rows_a = len(a)
    cols_a = len(a[0])
    rows_b = len(b)
    cols_b = len(b[0])
    if cols_a != rows_b:
        print('Number of A columns must equal number of B rows.')
    new_matrix = []
    while len(new_matrix) < rows_a:  # while len small the len rows
        new_matrix.append([])  # add place
        while len(new_matrix[-1]) < cols_b:
            new_matrix[-1].append(0.0)  # add value
    for i in range(rows_a):
        for j in range(cols_b):
            total = 0
            for k in range(cols_a):
                total += a[i][k] * b[k][j]  # mul mat
            new_matrix[i][j] = total
    return new_matrix  # return the A*B=new matrix


def create_i(matrix):  # A function that creates and returns the unit matrix
    i_mtrx = list(range(len(matrix)))  # make it list
    for i in range(len(i_mtrx)):
        i_mtrx[i] = list(range(len(i_mtrx)))

    for i in range(len(i_mtrx)):
        for j in range(len(i_mtrx[i])):
            i_mtrx[i][j] = 0.0  # put the zero

    for i in range(len(i_mtrx)):
        i_mtrx[i][i] = 1.0  # put the pivot
    return i_mtrx  # unit matrix


def create_i_by_size(size):
    mtrx = []
    for i in range(size):
        mtrx.append([])
        for j in range(size):
            mtrx[i].append(0)
    for i in range(size):
        mtrx[i][i] = 1
    return mtrx


def swap_rows(matrix,row_1,row_2,size):
    i_matrix = create_i_by_size(size)
    i_matrix[row_1],i_matrix[row_2] = i_matrix[row_2],i_matrix[row_1]
    return matrix_multiply(i_matrix,matrix)


def inverse(matrix):  # A function that creates and returns the inverse matrix to matrix A
    new_matrix = create_i(matrix)  # Creating the unit matrix
    count = 0
    check = False  # flag
    while count <= len(matrix) and check == False:
        if matrix[count][0] != 0:  # if the val in place not 0
            check = True  # flag
        count = count + 1  # ++
    if not check:
        print("ERROR")
    else:
        temp = matrix[count - 1]
        matrix[count - 1] = matrix[0]  # put zero
        matrix[0] = temp
        temp = new_matrix[count - 1]
        new_matrix[count - 1] = new_matrix[0]
        new_matrix[0] = temp

        for x in range(len(matrix)):
            divider = matrix[x][x]  # find the div val
            if divider == 0:
                divider = 1
            for i in range(len(matrix)):
                matrix[x][i] = matrix[x][i] / divider  # find the new index
                new_matrix[x][i] = new_matrix[x][i] / divider
            for row in range(len(matrix)):
                if row != x:
                    divider = matrix[row][x]
                    for i in range(len(matrix)):
                        matrix[row][i] = matrix[row][i] - divider * matrix[x][i]
                        new_matrix[row][i] = new_matrix[row][i] - divider * new_matrix[x][i]
    return new_matrix  # Return of the inverse matrix


def null_this(matrix, x, y, size, pivot):
    i_matrix = create_i_by_size(size)
    return matrix_multiply(edit_this(i_matrix, x, y, -1*matrix[x][y]/pivot), matrix)


def set_pivot(matrix,a,b,size):
    i_matrix = create_i_by_size(size)
    return matrix_multiply(edit_this(i_matrix, a, b, 1/matrix[a][b]), matrix)


def edit_this(matrix, a, b, value):
    matrix[a][b] = value
    return matrix


def solve_this(matrix,size):
    for i in range(size):
        pivot = abs(matrix[i][i])
        max_rox = i
        row = i+1
        while row<size:
            if abs(matrix[row][i] > pivot):
                pivot = abs(matrix[row][i])
                max_rox = row
            row += 1
        matrix = swap_rows(matrix, i, max_rox, size)
        matrix = set_pivot(matrix, i, i, size)
        row = i + 1
        while row < size:
            matrix = null_this(matrix, row, i, size, matrix[i][i])
            row += 1
    for i in range(1, size):
        row = i - 1
        while row >=0:
            matrix = null_this(matrix, row, i, size, matrix[i][i])
            row -= 1
    return matrix

#####################################################################################

# Interpolation methods


def linear_interpolation(requested_p):

    # Range for the number requested
    x1 = range_creator(requested_p)
    x2 = x1 + 1

    equation_part1 = ((yVectors[x1] - yVectors[x2]) / (xVectors[x1] - xVectors[x2])) * requested_p
    equation_part2 = (yVectors[x2]*xVectors[x1]-yVectors[x1]*xVectors[x2]) / (xVectors[x1]-xVectors[x2])
    val = equation_part1 + equation_part2
    print("f(%.2f) =" % requested_p, val)


def polynomial_interpolation(points, requested_p):
    # creating a new matrix
    mat = list(range(len(points)))
    for i in range(len(mat)):
        mat[i] = list(range(len(mat)))
    for row in range(len(points)):
        mat[row][0] = 1
    for row in range(len(points)):
        for col in range(1, len(points)):
            mat[row][col] = pow(points[row][0], col)
    res_mat = list(range(len(points)))
    for i in range(len(res_mat)):
        res_mat[i] = list(range(1))
    for row in range(len(res_mat)):
        res_mat[row][0] = points[row][1]
    vector_a = matrix_multiply(inverse(mat), res_mat)
    print('a[0]->a[%.f] =' % (len(points)-1), vector_a)
    sum1 = 0
    for i in range(len(vector_a)):
        if i == 0:
            sum1 = vector_a[i][0]
        else:
            sum1 += vector_a[i][0] * requested_p ** i
    print('P%.f(%.2f) = %.10f' % (len(points) - 1, requested_p, sum1))


def lagrange_interpolation(requested_x):
    m = len(xVectors)
    n = m-1
    yp = 0
    for i in range(n+1):
        p = 1
        for j in range(n+1):
            if j != i:
                p *= (requested_x - xVectors[j]) / (xVectors[i] - xVectors[j])
        yp += yVectors[i]*p
    print('Pn(%.2f) = %f' % (requested_x, yp))


def neville_interpolation(requested_p):
    n = len(xVectors)
    p = n*[0]
    for k in range(n):
        for i in range(n-k):
            if k == 0:
                p[i] = yVectors[i]
            else:
                p[i] = ((requested_p - xVectors[i + k]) * p[i] +
                        (xVectors[i] - requested_p) * p[i + 1]) / \
                       (xVectors[i] - xVectors[i + k])
    print("p(x=%.2f) =" % requested_p, p[0])


def cubic_spline_interpolation(requested_p, x_vectors, y_vectors):
    gamma = []
    h = []
    d = []
    m_u = []

    for i in range(0, len(x_vectors) - 1):
        h.append(x_vectors[i + 1] - x_vectors[i])
    for i in range(1, len(x_vectors) - 1):
        gamma.append(h[i]/(h[i] + h[i - 1]))
        m_u.append(1 - h[i]/(h[i] + h[i - 1]))
        d.append((6 / (h[i] + h[i - 1]) * ((y_vectors[i + 1] - y_vectors[i]) / h[i] - (y_vectors[i] - y_vectors[i - 1]) / h[i - 1])))

    mtrx = create_i_by_size(len(d))
    for i in range(len(d)):
        mtrx[i][i] = 2
        if i != 0:
            mtrx[i][i-1] = m_u[i]
        if i != len(d)-1:
            mtrx[i][i+1] = gamma[i]
        mtrx[i].append(d[i])

    m = [0]
    result = solve_this(mtrx,len(d))
    for x in range(len(result) - 1):
        m.append(result[x][-1])
    m.append(0)

    for y in range(len(y_vectors) - 1):
        if requested_p > x_vectors[y]:
            if requested_p < x_vectors[y + 1]:
                 print("f(x)=", ((x_vectors[y + 1] - requested_p) ** 3 * m[y] + (requested_p - x_vectors[y]) ** 3 * m[y + 1]) / (6 * h[y]) \
                       + ((x_vectors[y+1] - requested_p) * y_vectors[y] + (requested_p - x_vectors[y]) * y_vectors[y + 1]) / (h[y]) \
                       - h[y] * ((x_vectors[y + 1] - requested_p) * m[y] + (requested_p - x_vectors[y]) * m[y + 1]) / 6)

#####################################################################################


# our table vectors
xVectors = [0, 1, 2, 3, 4, 5, 6]
yVectors = [0, 0.8415, 0.9093, 0.1411, -0.7568, -0.9589, -0.2794]
ourVectorsAsPoints = [[0, 0], [1, 0.8415], [2, 0.9093], [3, 0.1411], [4, -0.7568], [5, -0.9589], [6, -0.2794]]


print("For these table vectors")
print("X(n)=", xVectors)
print("Y(n)=", yVectors)
print("###linear interpolation###")
linear_interpolation(2.5)
print("###polynmial interpolation###")
polynomial_interpolation(ourVectorsAsPoints, 2.5)
print("###lagrange interpolation###")
lagrange_interpolation(2.5)
print("###neville interpolation###")
neville_interpolation(2.5)
print("###cubic spline interpolation###")
cubic_spline_interpolation(2.5, xVectors, yVectors)
