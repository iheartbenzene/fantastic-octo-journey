import os
import psycopg2

DATABASE_URL = os.environ['postgresql-vertical-29624']

conn = psycopg2.connet(DATABASE_URL, sslmode = 'require')
