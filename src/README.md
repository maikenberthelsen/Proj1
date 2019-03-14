# COM402 Homework 1  

## Exercise 1 & 2
Explanantions of exercise 1 and 2 can be found in the word document(due to the picture)


## Exercise 3 & 4 
Go to the folder where you have interceptor.py, interceptor2.py and run_dockers.sh. 
Start Docker.

## Exercise 3
In this exercise I use interceptor.py

I start by typing: sh run_dockers.sh maiken.berthelsen@epfl.ch
Next command: docker exec -it attacker python3 shared/interceptor.py

You will maybe have to wait a while, but will eventuelly receive 

{"product": "Lq3Sos+rCh6M1+nj6lQoxrrVn/zLAZGAZFy6Pg62+Nk=", "shipping_address": "lausanne"}
TBt1t2Q5RvOt96EwDTIZA33e7tmG+e1BzBQitwW3GmY=
The last string is the token for exercise 3.


## Exercise 4
Type docker exec -it attacker python3 shared/interceptor2.py
All the secrets are printed once they are found, and in the end this is printed:

{'secrets': ['1458.9681.0437.2905', '>D3?R>?LGE>2JH', '8946/1123/3234/2400', '6358/6583/0361/2674', 'XQNEV5E3L<>Q8'], 'student_email': 'maiken.berthelsen@epfl.ch'}
Y9sPsTQS789+LoTf8u/wWJhTAqwFCz/lR9PJUX84paw=
The last string is the token for exercise 4.

