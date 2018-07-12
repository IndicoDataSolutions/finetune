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
        sh 'CUDA_VISIBLE_DEVICES="" nosetests -sv --nologcapture'
      }
    }
  }
}