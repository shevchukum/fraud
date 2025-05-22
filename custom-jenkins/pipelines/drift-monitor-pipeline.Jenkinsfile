pipeline {
    agent any
    triggers {
        cron('H 0 * * *') 
    }
    stages {
        stage('Check Drift') {
            steps {
                script {
                    def driftStatus = sh(
                        script: 'python3 /var/drift_scripts/drift-checker.py',
                        returnStatus: true
                    )
                    if (driftStatus == 1) {
                        echo "✅ Drift detected! Starting retraining..."
                        build job: 'mlflow-training-deploy', wait: true
                    } else if (driftStatus == 0) {
                        echo "👍 No drift. Keep existing model."
                    } else {
                        error "⚠️ Drift checker failed! Possibly too few new data."
                    }
                }
            }
        }
    }
}
