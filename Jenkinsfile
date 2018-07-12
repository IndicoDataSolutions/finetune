pipeline {
  agent any
  stages {
    stage('error') {
      steps {
        sh './docker/build_docker.sh '
      }
    }
  }
}