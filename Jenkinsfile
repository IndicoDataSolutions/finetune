pipeline {
  agent any
  stages {
    stage('Build Docker Image') {
      steps {
        sh 'echo $USER'
        sh 'docker container rm -f finetune || true'
        sh './docker/build_docker.sh '
      }
    }
    stage('Start Docker Image') {
      steps {
        sh './docker/start_docker.sh'
      }
    }
<<<<<<< HEAD
    stage('Test') {
      steps {
        sh 'docker exec finetune nosetests -sv --nologcapture --with-xunit'
      }

      post {
        always {
          junit "**/nosetests.xml"
        }
=======
    stage('Run Tests ') {
      steps {
        sh 'docker exec finetune nosetests -sv --nologcapture'
>>>>>>> df45c2a... Added Jenkinsfile
      }
    }
    stage('Remove container') {
      steps {
        sh 'docker rm -f finetune'
      }
    }
  }
  post { 
    always { 
      cleanWs()
    }
  }
}