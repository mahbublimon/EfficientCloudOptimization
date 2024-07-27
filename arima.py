from sagemaker import get_execution_role
import sagemaker
from sagemaker.amazon.amazon_estimator import get_image_uri

role = get_execution_role()
sess = sagemaker.Session()
bucket = 'my-monitoring-data'
prefix = 'sagemaker/arima'

s3_input_train = sagemaker.s3_input(s3_data=f's3://{bucket}/{prefix}/train/', content_type='text/csv')
s3_input_test = sagemaker.s3_input(s3_data=f's3://{bucket}/{prefix}/test/', content_type='text/csv')

container = get_image_uri(sess.boto_region_name, 'forecasting-deepar', 'latest')

estimator = sagemaker.estimator.Estimator(
    container,
    role,
    train_instance_count=1,
    train_instance_type='ml.c4.xlarge',
    output_path=f's3://{bucket}/{prefix}/output',
    sagemaker_session=sess
)

estimator.fit({'train': s3_input_train, 'test': s3_input_test})