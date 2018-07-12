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
        sh './scripts/build_docker.sh '
      }
    }
  }
}