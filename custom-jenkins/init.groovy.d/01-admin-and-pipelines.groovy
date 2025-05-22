import jenkins.model.*
import hudson.security.*
import org.acegisecurity.context.SecurityContextHolder
import org.jenkinsci.plugins.workflow.job.WorkflowJob
import org.jenkinsci.plugins.workflow.cps.CpsFlowDefinition
import hudson.triggers.TimerTrigger

def instance = Jenkins.getInstance()
def adminUsername = System.getenv("JENKINS_ADMIN_ID") ?: "admin"
def adminPassword = System.getenv("JENKINS_ADMIN_PASSWORD") ?: "admin"

// Creating admin user
def user = hudson.model.User.get(adminUsername, false)
if (user == null || user.isUnknown()) {
    println "--> Creating admin user"
    def hudsonRealm = new HudsonPrivateSecurityRealm(false)
    hudsonRealm.createAccount(adminUsername, adminPassword)
    instance.setSecurityRealm(hudsonRealm)

    def strategy = new FullControlOnceLoggedInAuthorizationStrategy()
    strategy.setAllowAnonymousRead(false)
    instance.setAuthorizationStrategy(strategy)
    instance.save()
} else {
    println "ℹ️ Admin user '${adminUsername}' already exists"
}

// Ayth under user=admin
def authentication = hudson.model.User.get(adminUsername).impersonate()
SecurityContextHolder.getContext().setAuthentication(authentication)

try {
    println "⏳ Sleeping 15 seconds before pipeline creation"
    sleep(15000)

    def pipelines = [
        [name: "drift-monitor-pipeline", jenkinsfilePath: "/var/jenkins_home/pipelines/drift-monitor-pipeline.Jenkinsfile", cronTrigger: "H 0 * * *"],
        [name: "monthly-retraining", jenkinsfilePath: "/var/jenkins_home/pipelines/monthly-retraining.Jenkinsfile", cronTrigger: "H 0 1 * *"],
        [name: "test-retraining", jenkinsfilePath: "/var/jenkins_home/pipelines/test-retraining.Jenkinsfile", cronTrigger: ""],
        [name: "mlflow-training-deploy", jenkinsfilePath: "/var/jenkins_home/pipelines/mlflow-training-deploy.Jenkinsfile", cronTrigger: ""]
    ]

    pipelines.each { pipeline ->
        def file = new File(pipeline.jenkinsfilePath)
        if (!file.exists() || !file.canRead()) {
            println "❌ Jenkinsfile not found: ${file.absolutePath}"
            return
        }

        def job = instance.getItem(pipeline.name)
        if (job == null) {
            job = instance.createProject(WorkflowJob, pipeline.name)
            job.definition = new CpsFlowDefinition(file.text, true)

            if (pipeline.cronTrigger) {
                job.addTrigger(new TimerTrigger(pipeline.cronTrigger))
            }

            job.save()
            println "✅ Created pipeline: ${pipeline.name}"
        } else {
            println "ℹ️ Pipeline already exists: ${pipeline.name}"
        }
    }

    println "✅ All pipelines handled!"
} finally {
    SecurityContextHolder.clearContext()
}
