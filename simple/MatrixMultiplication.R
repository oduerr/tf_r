---
title: "Simple matrix multiplication"
author: "Oliver Dürr"
date: "11/7/2016"
output: github_document
---

library(tensorflow)

m1_values = matrix(c(3,3), nrow = 1)
m1_values = matrix(c(2,2), nrow = 2)

m1 = tf$constant([[3., 3.]], name='M1')
m2 = tf$constant([[2.],[2.]], name='M2')
product = 10*tf.matmul(m1,m2)