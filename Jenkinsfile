pipeline {
  agent {
    dockerfile {
      filename 'docker/Dockerfile'
    }

  }
  stages {
    stage('error') {
      steps {
        echo 'Running tests...'
        sh 'nosetests'
      }
    }
  }
}