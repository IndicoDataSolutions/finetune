pipeline {
  agent any
  stages {
    stage('Build Docker Image') {
      steps {
        sh './docker/build_docker.sh '
      }
    }
    stage('Start Docker Image') {
      steps {
        sh './docker/start_docker.sh'
      }
    }
  }
}