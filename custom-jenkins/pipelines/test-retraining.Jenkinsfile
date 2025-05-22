pipeline {
    agent any
    stages {
        stage('Test Retrain') {
            steps {
                script {
                    def dates = ['2025-05-01', '2025-06-01', '2025-07-01', '2025-08-01', '2025-09-01', '2025-10-01',
                                 '2025-11-01', '2025-12-01', '2026-01-01', '2026-02-01', '2026-03-01', '2026-04-01']
                    
                    for (date in dates) {
                        echo "Launching retrain job for date: ${date}"
                        try {
                            build job: 'mlflow-training-deploy', 
                                  parameters: [string(name: 'TODAY', value: date)],
                                  wait: true
                        } catch (err) {
                            echo "Failed to trigger job for date ${date}: ${err}"
                        }
                    }
                }
            }
        }
    }
}