pipeline {
    agent any
    triggers {
        cron('H 0 1 * *') 
    }
    parameters {
        string(name: 'TODAY', defaultValue: '', description: 'Until which date data to be used')
    }
    stages {
        stage('Train model') {
            steps {
                script {
                    def trainOutput = sh(
                        script: 'docker-compose -f /var/jenkins_home/docker-compose.train.yml -p myproject run --rm -e TODAY=$TODAY mlflow-train 2>&1',
                        returnStdout: true
                    ).trim()
                    
                    echo trainOutput
                    
                    // Checking the result message
                    if (trainOutput.contains("Promoted new model")) {
                        env.MODEL_PROMOTED = "true"
                    } else {
                        env.MODEL_PROMOTED = "false"
                    }
                }
            }
        }
        
        stage('Deploy to FastAPI') {
            when {
                expression { env.MODEL_PROMOTED == "true" }
            }
            steps {
                sh 'curl -X POST http://fastapi:8000/update-model'
            }
        }
    }
}
