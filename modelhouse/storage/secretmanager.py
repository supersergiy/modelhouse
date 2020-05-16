'''
Code mostly taken from github.com/seung-lab/cloud-volume by Will Silversmith
'''

from __future__ import print_function
from collections import defaultdict
import os
import json
import warnings

from google.oauth2 import service_account


class SecretManager():
    def __init__(self):

        HOME = os.path.expanduser('~')
        CLOUDVOLUME_DIR = os.path.join(HOME, '.cloudvolume')
        self.secrets_folder = os.path.join(CLOUDVOLUME_DIR, "secrets")

        self.project_name = self.get_default_google_project_name()
        self.google_credentials_cache = {}
        #self.google_credentials_path = get_secretfile_path('google-secret.json')

        self.aws_credentials_cache = defaultdict(dict)
        #self.aws_credentials_path = secretpath('secrets/aws-secret.json')

    def set_secrets_folder(self, secrets_folder):
        self.secrets_folder = secrets_folder

        self.project_name = self.get_default_google_project_name()
        self.google_credentials_cache = {}
        #self.google_credentials_path = get_secretfile_path('google-secret.json')

        self.aws_credentials_cache = defaultdict(dict)
        #self.aws_credentials_path = secretpath('secrets/aws-secret.json')


    def get_secretfile_path(self, secretfile):
        return os.path.join(self.secrets_folder, secretfile)


    def get_default_google_project_name(self):
        if 'GOOGLE_PROJECT_NAME' in os.environ:
            return os.environ['GOOGLE_PROJECT_NAME']
        else:
            default_credentials_path = self.get_secretfile_path(
                    'google-secret.json')
            if os.path.exists(default_credentials_path):
                with open(default_credentials_path, 'rt') as f:
                    return json.loads(f.read())['project_id']

        return None


    def get_google_credentials(self, bucket=None):
        if bucket in self.google_credentials_cache.keys():
            return self.google_credentials_cache[bucket]
        paths = []
        paths.append(self.get_secretfile_path('google-secret.json'))
        if bucket is not None:
            bucket_secretfile = '{}-google-secret.json'.format(bucket)
            paths.append(self.get_secretfile_path(bucket_secretfile))

        google_credentials = None
        project_name = self.project_name
        for google_credentials_path in paths:
            if os.path.exists(google_credentials_path):
                google_credentials = service_account.Credentials \
                    .from_service_account_file(google_credentials_path)

                with open(google_credentials_path, 'rt') as f:
                    project_name = json.loads(f.read())['project_id']
                break

        if google_credentials == None:
            warnings.warn('Using default Google credentials. \
                    There is no {} google-secret.json set.'.format(secrets_folder))
        else:
            self.google_credentials_cache[bucket] = \
                    (project_name, google_credentials)

        return project_name, google_credentials

    def aws_credentials(self, bucket=None, service='aws'):
        if service == 's3':
            service = 'aws'

        if bucket in self.aws_credentials_cache.keys():
            return self.aws_credentials_cache[bucket]


        paths = []
        default_secret_path = '{}-secret.json'.format(service)
        paths.append(self.get_secretfile_path(default_secret_path))
        if bucket is not None:
            bucket_secretfile = '{}-{}-secret.json'.format(bucket,
                service)
            paths.append(self.get_secretfile_path(bucket_secretfile))

        aws_credentials = {}
        aws_credentials_path = secretpath(default_file_path)
        for aws_credentials_path in paths:
            if os.path.exists(aws_credentials_path):
                with open(aws_credentials_path, 'r') as f:
                    aws_credentials = json.loads(f.read())
                break

        if not aws_credentials:
            # did not find any secret json file, will try to find it in environment variables
            if 'AWS_ACCESS_KEY_ID' in os.environ and 'AWS_SECRET_ACCESS_KEY' in os.environ:
                aws_credentials = {
                    'AWS_ACCESS_KEY_ID': os.environ['AWS_ACCESS_KEY_ID'],
                    'AWS_SECRET_ACCESS_KEY': os.environ['AWS_SECRET_ACCESS_KEY'],
                }
            if 'AWS_DEFAULT_REGION' in os.environ:
                aws_credentials['AWS_DEFAULT_REGION'] = os.environ['AWS_DEFAULT_REGION']

        self.aws_credentials_cache[service][bucket] = aws_credentials
        return aws_credentials

secret_manager = SecretManager()
