# COM402 Homework 2  

## Exercise 1 
To run:
- python ex1.py

In another terminal I wrote this in order to debug my program:
- curl -d '{"user":"maiken.berthelsen@epfl.ch", "pass":"IwQfDhdOXQcLFlQJRQQGCA8uRQQJTEoMSA=="}' -H "Content-Type: application/json" -X POST http://127.0.0.1:5000/hw2/ex1

## Exercise 2
May need to install websockets
- pip install websockets

To run:
- python ex2.py

The token will then be printed.

## Exercise 3 
To run:
- python ex2.py

In a new terminal I wrote this to debug:
- curl -d '{"user":"maiken.berthelsen@epfl.ch", "pass":"IwQfDhdOXQcLFlQJRQQGCA8uRQQJTEoMSA=="}' -H "Content-Type: application/json" -X POST http://127.0.0.1:5000/ex3/login

**When uploading to the verification site I sometimes received and error and sometimes a token with the same script. So if an error occurs please try to upload it several times as it clearly works given that I got the token and I did not find out why it sometimes outputted an error.**



## Exercise 4a
Typed the following commands:
- docker start -i hw2_ex4
- nginx

Create key and certificate, have already created the folder /etc/nginx/ssl
- sudo openssl req -x509 -nodes -days 365 -newkey rsa:2048 -keyout /etc/nginx/ssl/localhost.key -out /etc/nginx/ssl/localhost.crt

Example fill in, common name should be localhost: 

- Country Name (2 letter code) [AU]:NO
- State or Province Name (full name) [Some-State]:server
- Locality Name (eg, city) []:Oslo
- Organization Name (eg, company) [Internet Widgits Pty Ltd]:SERVER
- Organizational Unit Name (eg, section) []:s
- Common Name (e.g. server FQDN or YOUR name) []:localhost
- Email Address []:maiken.berthelsen@epfl.ch

Change the location in the default.conf file to 
- ssl_certificate /etc/nginx/ssl/localhost.crt;
- ssl_certificate_key /etc/nginx/ssl/localhost.key;

In order to check that the config-file is ok type
- sudo nginx -t

To reload nginx 
- sudo nginx -s reload

I exited the program and then started it again, and wrote nginx, then ./verify.sh a

(The index.htms filed has also been changed by using vim)

## Exercise 4b
Typed the following commands:
- docker start -i hw2_ex4
- nginx

Inside the /etc/nginx/ssl folder

Creating the servers private key, need to fill in a password
- openssl genrsa -des3 -out server.key 1024

Create the certificate signing request, need to fill in the same password
- openssl req -new -key server.key -out server.csr

Example of what I filled in, can be filled in with whatever as long as common name is localhost

- Country Name (2 letter code) [AU]:NO
- State or Province Name (full name) [Some-State]:server
- Locality Name (eg, city) []:Oslo
- Organization Name (eg, company) [Internet Widgits Pty Ltd]:SERVER
- Organizational Unit Name (eg, section) []:s
- Common Name (e.g. server FQDN or YOUR name) []:localhost
- Email Address []:maiken.berthelsen@epfl.ch

Removing the need to fill in the password for starting up NGINX with SSL
- cp server.key server.key.org
- openssl rsa -in server.key.org -out server.key

Copy server.csr to the COM402's CA, and receive the certificate.

Copy the certificate into a file called server.crt.

In default.conf change the name where you can find the ssl_certificate and ssl_certificate_key to:
- ssl_certificate /etc/nginx/ssl/localhost.crt;
- ssl_certificate_key /etc/nginx/ssl/localhost.key;

In order to check that the config-file is ok type
- sudo nginx -t

To reload nginx 
- sudo nginx -s reload

I exited the program and then started it again, and wrote nginx, then ./verify.sh b




