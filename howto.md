# Как заставить работать

Скопируйте sitecustomize.py в site-packages, а сертификаты — в Документы.

Допом нужно настроить переменные для pip:
```
REQUESTS_CA_BUNDLE
SSL_CERT_FILE
```
в Powershell:
```
[Environment]::SetEnvironmentVariable("REQUESTS_CA_BUNDLE", "$env:USERPROFILE\Documents\certs.pem", "User")
[Environment]::SetEnvironmentVariable("SSL_CERT_FILE", "$env:USERPROFILE\Documents\certs.pem", "User")
```