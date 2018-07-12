pipeline {
  agent any
  stages {
    stage('error') {
      steps {
        sh './scripts/build_docker.sh '
        sh 'ls'
      }
    }
  }
}