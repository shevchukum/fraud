pipeline {
  agent any
  triggers {
    cron('H 0 1 * *') 
  }

  stages {
    stage('Retrain Model') {
        steps {
            build job: 'mlflow-training-deploy', wait: true
        }
    }
  }
}