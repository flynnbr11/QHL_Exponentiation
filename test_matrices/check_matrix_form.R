a_00 = 0 
a_01 = 0
a_02 = 1 + 1i
a_10 = 0
a_11 = 0
a_12 = 0
a_20 = Conj(a_02)
a_21 = 0
a_22 = 0


vals=c(a_00,a_01,a_02, a_10,a_11,a_12, a_20,a_21,a_22)
size=3



H = matrix(data=vals, nrow=size, ncol=size, byrow=TRUE)
squared = H %*% H
cubed = squared %*%H
fourth=squared%*%squared

print(H)
print(squared)
print(cubed)