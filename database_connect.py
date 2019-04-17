import os
import psycopg2

DATABASE_URL = os.environ['DATABASE_URL']

conn = psycopg2.connet(DATABASE_URL, sslmode = 'require')
