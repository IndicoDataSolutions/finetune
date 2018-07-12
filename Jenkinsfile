pipeline {
  agent any
  stages {
    stage('error') {
      steps {
        echo 'Running tests...'
        sh './scripts/build_docker.sh '
      }
    }
  }
}