pipeline {
  agent {
    docker {
      image 'indico/finetune'
    }

  }
  stages {
    stage('') {
      steps {
        echo 'Running tests...'
        sh 'nosetests'
      }
    }
  }
}